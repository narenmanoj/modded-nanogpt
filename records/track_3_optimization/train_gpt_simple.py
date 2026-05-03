"""
train_gpt_simple.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.
"""

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import argparse
import uuid
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.distributed as dist


########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (norm(x.float()) * self.gains).type_as(x)

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim//4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


########################################
#              Optimizer               #
########################################

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95,
                 lookahead_alpha=0.0, lookahead_mode="deterministic"):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        assert lookahead_mode in ("deterministic", "gaussian")
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu,
                        lookahead_alpha=lookahead_alpha, lookahead_mode=lookahead_mode)
        super().__init__(params, defaults)

    @torch.no_grad()
    def un_shift_inplace(self):
        """Subtract each param's lookahead_delta in place and all-gather, so the live params
        become W_t (un-shifted) on every rank. Pair with step(skip_unshift=True) to avoid
        double-un-shifting. Used to run a no_grad forward at the unperturbed weights for diagnostics."""
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            if group["lookahead_alpha"] == 0.0:
                continue
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if "lookahead_delta" in state:
                        p.sub_(state["lookahead_delta"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])

    @torch.no_grad()
    def step(self, skip_unshift=False):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            alpha = group["lookahead_alpha"]
            mode = group["lookahead_mode"]
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                        state["lookahead_delta"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                    # Un-shift: grad was evaluated at W_t + (perturbation); recover W_t.
                    # skip_unshift=True when caller already un-shifted via un_shift_inplace().
                    if alpha != 0.0 and not skip_unshift:
                        p.sub_(state["lookahead_delta"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                    # Store new shift; gathered value is the lookahead/perturbed point for next step's forward.
                    if alpha != 0.0:
                        if mode == "deterministic":
                            # Optimism (alpha>0) / pessimism (alpha<0) along the previous update direction.
                            state["lookahead_delta"].copy_(update).mul_(-alpha * group["lr"])
                        else:  # gaussian
                            # i.i.d. N(0, s^2) with s chosen so E[||xi||_op] ≈ |alpha| * ||p||_op:
                            # the perturbation magnitude is alpha times the current weight's spectral
                            # norm, so alpha is a unit-free fraction of the weight scale (no LR mixing).
                            # Cost: one SVD per Muon-managed param per step.
                            weight_op = torch.linalg.matrix_norm(p.float(), ord=2)
                            m, n = p.shape[-2], p.shape[-1]
                            state["lookahead_delta"].normal_(0.0, 1.0)
                            state["lookahead_delta"].mul_(weight_op * (alpha / (m ** 0.5 + n ** 0.5)))
                        p.add_(state["lookahead_delta"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


########################################
#                Setup                 #
########################################

# torchrun sets these env variables
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
# this code can be run equivalently with 1, 2, 4, or 8 gpus.
assert 8 % dist.get_world_size() == 0

# logging setup
run_id = str(uuid.uuid4())
writer = None
if dist.get_rank() == 0:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    hparams_file = f"logs/{run_id}_hyperparameters.txt"
    open(hparams_file, "w").close()  # truncate so multiple log_hparam calls just append
    print(logfile)
def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)
def log_hparam(s):
    """Log to main logfile + console + a dedicated hyperparameters file (rank 0 only)."""
    print0(s, console=True)
    if dist.get_rank() == 0:
        with open(hparams_file, "a") as f:
            print(s, file=f)

# we begin by logging this file itself
print0(code)
print0("="*100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0(f"Running on device_name={torch.cuda.get_device_name(device)} with world_size={dist.get_world_size()}")
print0("="*100)

val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024
mbs = 64
train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)


########################################
#       Init & Optim Hyperparams       #
########################################

# CLI args for sweeping. Logged below so the chosen values are recoverable from the logfile.
parser = argparse.ArgumentParser()
parser.add_argument("--lookahead_alpha", type=float, default=0.0,
                    help="Perturbation strength for Muon. 0 = standard Muon. "
                         "alpha=1 shifts the gradient query point by an amount whose operator "
                         "norm equals (or, for gaussian mode, has expected operator norm equal to) "
                         "||x_t - x_{t-1}||_op.")
parser.add_argument("--lookahead_mode", choices=("deterministic", "gaussian"),
                    default="deterministic",
                    help="deterministic: shift along previous update direction (optimism if alpha>0, "
                         "pessimism if alpha<0). gaussian: i.i.d. Gaussian shift, scaled so that at "
                         "alpha=1 the expected spectral norm matches the previous step's spectral norm.")
args = parser.parse_args()
log_hparam(f"cli_args: {vars(args)}")

# build a human-readable tensorboard run name encoding the perturbation regime
def _run_tag(alpha: float, mode: str) -> str:
    if alpha == 0.0:
        return "baseline"
    if mode == "gaussian":
        return f"gaussian_a_{alpha:g}"
    return (f"optimism_a_{alpha:g}" if alpha > 0 else f"pessimism_a_{abs(alpha):g}")
if dist.get_rank() == 0:
    run_tag = _run_tag(args.lookahead_alpha, args.lookahead_mode)
    writer = SummaryWriter(log_dir=f"runs/{run_tag}_{run_id[:8]}")
    log_hparam(f"tensorboard run dir: runs/{run_tag}_{run_id[:8]}")

# we want to minimize this while still reaching 3.28 val loss
train_steps = 7000

# initialize model parameters
for name, p in model.named_parameters():
    if "proj" in name:
        p.data.zero_()

# create the optimizer(s)
optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                    dict(params=[model.proj.weight], lr=1/320),
                    dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                   betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2],
                  lr=0.025, weight_decay=0.0125,
                  lookahead_alpha=args.lookahead_alpha,
                  lookahead_mode=args.lookahead_mode)
optimizers = [optimizer1, optimizer2]
assert set(p for opt in optimizers for group in opt.param_groups
           for p in group["params"]) == set(model.parameters())
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# Map params to names for per-tensor tensorboard logging.
name_by_param = {p: n for n, p in model.named_parameters()}

# Log resolved optimizer hyperparameters so each run is self-describing.
log_hparam(f"train_steps: {train_steps}")
for opt_name, opt in [("optimizer1", optimizer1), ("optimizer2", optimizer2)]:
    log_hparam(f"{opt_name}: {opt.__class__.__name__}")
    for i, group in enumerate(opt.param_groups):
        hp = {k: v for k, v in group.items() if k != "params"}
        log_hparam(f"  group[{i}] (n_params={len(group['params'])}): {hp}")

# learning rate schedule: stable then decay
def set_hparams(step, cooldown_frac=0.7):
    progress = step / train_steps
    assert 0 <= progress < 1
    if progress < 1 - cooldown_frac:
        eta = 1.0
    else:
        eta = (1 - progress) / cooldown_frac
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * eta


########################################
#        Training and Validation       #
########################################

for p in model.parameters():
    dist.broadcast(p.detach(), 0)
# start the clock
training_time = 0
dist.barrier()
t0 = time.perf_counter()
for step in range(train_steps + 1):

    # --------------- VALIDATION SECTION -----------------
    if step == train_steps or step % 125 == 0:
        # stop the clock
        dist.barrier()
        training_time += time.perf_counter() - t0
        model.eval()
        val_loss = 0
        with torch.no_grad():
            assert len(val_inputs) % mbs == 0
            for i in range(len(val_inputs) // mbs):
                val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_loss /= val_tokens
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
               + f" step_avg:{1000*training_time/max(step, 1):.2f}ms", console=True)
        if writer is not None:
            writer.add_scalar("loss/val", val_loss.item(), step)
            writer.flush()
        model.train()
        # start the clock again
        dist.barrier()
        t0 = time.perf_counter()

    if step == train_steps:
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    # accumulate across microbatches in case we are running with fewer than 8 gpus
    assert len(inputs) % mbs == 0
    # Perturbed forward+backward: live Muon params are W̃_t = W_t + (perturbation), so this
    # produces the loss that the optimizer's gradient is evaluated against.
    local_loss_perturbed = torch.zeros((), device="cuda")
    for i in range(len(inputs) // mbs):
        loss = model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
        loss.backward()
        local_loss_perturbed += loss.detach()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)

    # Unperturbed forward (no_grad): un-shift Muon params to W_t and re-run forward on the same
    # microbatches. The optimizer.step() call below uses skip_unshift=True so it won't re-un-shift.
    local_loss_unperturbed = None
    if args.lookahead_alpha != 0.0:
        optimizer2.un_shift_inplace()
        local_loss_unperturbed = torch.zeros((), device="cuda")
        with torch.no_grad():
            for i in range(len(inputs) // mbs):
                local_loss_unperturbed += model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])

    # gather train-loss + global grad norm before the optimizer mutates grads/params
    if writer is not None:
        dist.all_reduce(local_loss_perturbed, op=dist.ReduceOp.SUM)
        train_loss_perturbed = local_loss_perturbed / batch_size
        if local_loss_unperturbed is not None:
            dist.all_reduce(local_loss_unperturbed, op=dist.ReduceOp.SUM)
            train_loss_unperturbed = local_loss_unperturbed / batch_size
        grad_norm = torch.stack([p.grad.detach().float().pow(2).sum()
                                 for p in model.parameters()]).sum().sqrt()
    # set optimization hyperparameters and take a step
    set_hparams(step)
    optimizer1.step()
    optimizer2.step(skip_unshift=local_loss_unperturbed is not None)
    model.zero_grad(set_to_none=True)
    if writer is not None:
        writer.add_scalar("loss/train_perturbed", train_loss_perturbed.item(), step + 1)
        if local_loss_unperturbed is not None:
            writer.add_scalar("loss/train_unperturbed", train_loss_unperturbed.item(), step + 1)
            writer.add_scalar("loss/perturbation_cost",
                              (train_loss_perturbed - train_loss_unperturbed).item(), step + 1)
        writer.add_scalar("grad_norm", grad_norm.item(), step + 1)
        writer.add_scalar("lr/muon", optimizer2.param_groups[0]["lr"], step + 1)
        writer.add_scalar("lr/adamw_embed", optimizer1.param_groups[0]["lr"], step + 1)
        writer.add_scalar("lr/adamw_proj", optimizer1.param_groups[1]["lr"], step + 1)
        writer.add_scalar("lr/adamw_scalars", optimizer1.param_groups[2]["lr"], step + 1)
        # Per-Muon-param perturbation size relative to current weights.
        # Muon shards param updates across ranks, so optimizer2.state on rank 0 only contains
        # the subset of params this rank actually updated — that's the slice we log.
        if args.lookahead_alpha != 0.0:
            with torch.no_grad():
                for p, pstate in optimizer2.state.items():
                    if "lookahead_delta" not in pstate:
                        continue
                    pname = name_by_param.get(p)
                    if pname is None:
                        continue
                    delta = pstate["lookahead_delta"]
                    delta_op = torch.linalg.matrix_norm(delta.float(), ord=2)
                    weight_op = torch.linalg.matrix_norm(p.float(), ord=2)
                    delta_fro = delta.float().norm()
                    weight_fro = p.float().norm()
                    writer.add_scalar(f"perturb/{pname}/delta_op_over_weight_op",
                                      (delta_op / (weight_op + 1e-12)).item(), step + 1)
                    writer.add_scalar(f"perturb/{pname}/delta_fro_over_weight_fro",
                                      (delta_fro / (weight_fro + 1e-12)).item(), step + 1)
    approx_training_time = training_time + (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
           + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True, log=False)

if writer is not None:
    writer.close()
dist.destroy_process_group()
