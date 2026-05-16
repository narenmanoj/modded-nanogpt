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


def momentum_update(direction: Tensor, momentum: Tensor, mu=0.95, nesterov=True) -> Tensor:
    """Same Nesterov-style momentum convention as muon_update, but without
    the matrix-sign / Newton-Schulz projection.
    """
    momentum.lerp_(direction, 1 - mu)
    return direction.lerp(momentum, mu) if nesterov else momentum


def row_l2_normalize(x: Tensor, eps: float = 1e-12) -> Tensor:
    """Row-wise vector sign: x[i] / ||x[i]||_2, with zero rows left at zero."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

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
    def step(self):
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
                    if alpha != 0.0:
                        p.sub_(state["lookahead_delta"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                    # Store new shift; gathered value is the lookahead/perturbed point for next step's forward.
                    if alpha != 0.0:
                        if mode == "deterministic":
                            # Optimism (alpha>0) / pessimism (alpha<0) along the previous update direction.
                            state["lookahead_delta"].copy_(update).mul_(-alpha * group["lr"])
                        else:  # gaussian
                            # i.i.d. N(0, s^2) with s chosen so E[||xi||_op] = lr * max(1, sqrt(m/n))
                            # at alpha=1, matching the deterministic shift's spectral norm.
                            m, n = p.shape[-2], p.shape[-1]
                            s = group["lr"] * max(1.0, (m / n) ** 0.5) / (m ** 0.5 + n ** 0.5)
                            state["lookahead_delta"].normal_(0.0, s).mul_(alpha)
                        p.add_(state["lookahead_delta"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


def _ridged_inv(K: Tensor, gamma: float, eps: float) -> Tensor:
    """(K + (gamma * tr(K)/n + eps) * I)^{-1} via Cholesky, with identity fallback
    if the factorization fails. Matches the formula and the defensive handling in
    https://github.com/zhehangdu/Newton-Muon (precond_ridge_mult, precond_eps,
    cholesky_ex + bad-row identity replacement)."""
    n = K.size(-1)
    ridge = gamma * K.diagonal().mean() + eps
    eye = torch.eye(n, device=K.device, dtype=K.dtype)
    Kr = K + ridge * eye
    L, info = torch.linalg.cholesky_ex(Kr, check_errors=False)
    if info.item() != 0:
        return eye  # Non-PD: fall back to identity (no preconditioning this round).
    return torch.cholesky_inverse(L)


class NewtonMuon(torch.optim.Optimizer):
    """
    Newton-Muon: Muon with right-preconditioning by the inverse of the input
    Gram matrix. From "The Newton-Muon Optimizer" (arXiv:2604.01472).

    For each 2D weight W (m, n) of an nn.Linear with input z in R^n, maintains
    K ≈ E[z z^T] (shape n x n) via EWMA on Z^T Z / N, refreshed every
    `refresh_k` steps, with diagonal ridge gamma * tr(K)/n. Right-preconditions
    G ← G K^{-1} before the standard Muon (NS + (m,n)-scaling) update.

    Activations are captured via forward pre-hooks on the supplied modules,
    gated on module.training so validation passes don't pollute K.
    """
    def __init__(self, params_and_modules, lr=0.025, weight_decay=0.0125, mu=0.95,
                 beta=0.95, gamma=0.2, eps=1e-8, refresh_k=32,
                 lookahead_alpha=0.0, lookahead_mode="deterministic"):
        assert isinstance(params_and_modules, list) and len(params_and_modules) >= 1
        assert lookahead_mode in ("deterministic", "gaussian")
        # Sort by param size for balanced DDP work assignment (matches Muon)
        pairs = sorted(params_and_modules, key=lambda pm: pm[0].size(), reverse=True)
        params = [p for p, _ in pairs]
        assert isinstance(params[0], torch.nn.Parameter)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu,
                        beta=beta, gamma=gamma, eps=eps, refresh_k=refresh_k,
                        lookahead_alpha=lookahead_alpha, lookahead_mode=lookahead_mode)
        super().__init__(params, defaults)
        # Hook-time state: only the rank that will eventually own a param accumulates
        # its activation statistics. Assignment matches the inner loop below
        # (param index i is processed by rank i % world_size).
        world_size = dist.get_world_size()
        my_rank = dist.get_rank()
        self._owns = {id(p): (i % world_size) == my_rank for i, p in enumerate(params)}
        self._zz_accum = {}   # id(param) -> (n, n) accumulator
        self._n_accum = {}    # id(param) -> int sample count
        self._hooks = []
        for p, m in pairs:
            self._hooks.append(m.register_forward_pre_hook(self._make_hook(p)))

    def _make_hook(self, param):
        key = id(param)
        def hook(module, args):
            if not module.training:
                return
            if not self._owns.get(key, False):
                return
            x = args[0].detach()
            x = x.reshape(-1, x.size(-1)).float()  # (N_local, n)
            n = x.size(-1)
            if key not in self._zz_accum:
                self._zz_accum[key] = torch.zeros(n, n, device=x.device, dtype=torch.float32)
                self._n_accum[key] = 0
            self._zz_accum[key].addmm_(x.T, x)
            self._n_accum[key] += x.size(0)
        return hook

    @torch.no_grad()
    def step(self):
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
                    n_in = p.shape[-1]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                        state["lookahead_delta"] = torch.zeros_like(p)
                        state["K"] = 1e-3 * torch.eye(n_in, device=p.device, dtype=torch.float32)
                        state["Kinv"] = _ridged_inv(state["K"], group["gamma"], group["eps"])
                        state["step_count"] = 0
                    state["step_count"] += 1
                    # Refresh K from the just-accumulated Z^T Z (one step's worth,
                    # matching reference behavior where precond_flag gates the hook
                    # to fire only on refresh steps).
                    if state["step_count"] % group["refresh_k"] == 0 and id(p) in self._zz_accum:
                        N = self._n_accum[id(p)]
                        if N > 0:
                            beta = group["beta"]
                            state["K"].mul_(beta).add_(self._zz_accum[id(p)] / N, alpha=1 - beta)
                            state["Kinv"] = _ridged_inv(state["K"], group["gamma"], group["eps"])
                    # Right-precondition gradient (fp32 matmul, then cast back)
                    g_precond = (p.grad.float() @ state["Kinv"]).to(p.grad.dtype)
                    update = muon_update(g_precond, state["momentum"], mu=group["mu"])
                    if alpha != 0.0:
                        p.sub_(state["lookahead_delta"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                    if alpha != 0.0:
                        if mode == "deterministic":
                            state["lookahead_delta"].copy_(update).mul_(-alpha * group["lr"])
                        else:  # gaussian
                            m, n = p.shape[-2], p.shape[-1]
                            s = group["lr"] * max(1.0, (m / n) ** 0.5) / (m ** 0.5 + n ** 0.5)
                            state["lookahead_delta"].normal_(0.0, s).mul_(alpha)
                        p.add_(state["lookahead_delta"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])
        # Reset Z^TZ accumulators at end of every step, so each refresh draws on
        # exactly one step's worth of activations (matches reference behavior).
        for k in self._zz_accum:
            self._zz_accum[k].zero_()
            self._n_accum[k] = 0


class RowNormPullback(torch.optim.Optimizer):
    """
    Row-norm + pullback optimizer.

    For a Linear layer y = z W^T, let A = dL/dy be the activation-output
    gradient and K = E[z z^T]. The sharp ||.||_{infty,2} derivation gives

        Delta y = -lr * rsgn(A),

    where rsgn normalizes every token row of A to unit l2 norm. The minimum
    Frobenius-norm pullback to the PyTorch weight W is

        Delta W = -lr * E[rsgn(A)^T z] K^{-1}.

    This class estimates E[rsgn(A)^T z] with a backward hook on each supplied
    Linear module, estimates K with the same input hooks/statistics used by
    NewtonMuon, and applies the pulled-back direction directly. Optional
    parameter-space momentum can be enabled with mu > 0; setting mu=0 recovers
    the literal one-step row-norm pullback direction.

    Important implementation note: in this manually-sharded DDP setup, only the
    rank that owns a parameter accumulates its local activation/output-gradient
    statistics, matching NewtonMuon's ownership pattern. That is a stochastic
    local estimate of the pullback statistics, while p.grad is still all-reduced
    by the training loop.
    """
    def __init__(self, params_and_modules, lr=0.025, weight_decay=0.0125, mu=0.0,
                 beta=0.95, gamma=0.2, eps=1e-8, refresh_k=1, row_eps=1e-12,
                 lookahead_alpha=0.0, lookahead_mode="deterministic"):
        assert isinstance(params_and_modules, list) and len(params_and_modules) >= 1
        assert lookahead_mode in ("deterministic", "gaussian")
        pairs = sorted(params_and_modules, key=lambda pm: pm[0].size(), reverse=True)
        params = [p for p, _ in pairs]
        assert isinstance(params[0], torch.nn.Parameter)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu,
                        beta=beta, gamma=gamma, eps=eps, refresh_k=refresh_k,
                        row_eps=row_eps, lookahead_alpha=lookahead_alpha,
                        lookahead_mode=lookahead_mode)
        super().__init__(params, defaults)
        self._row_eps = row_eps

        world_size = dist.get_world_size()
        my_rank = dist.get_rank()
        self._owns = {id(p): (i % world_size) == my_rank for i, p in enumerate(params)}

        # Accumulators for the rank-local estimate of
        #   K_num = sum z_i z_i^T,
        #   RZ_num = sum rsgn(a_i)^T z_i.
        self._zz_accum = {}   # id(param) -> (n, n)
        self._rz_accum = {}   # id(param) -> (m, n)
        self._n_accum = {}    # id(param) -> int row/token count
        self._hooks = []
        for p, m in pairs:
            # Need a forward hook rather than a pre-hook because we attach a
            # tensor hook to the Linear output to see dL/dy during backward.
            self._hooks.append(m.register_forward_hook(self._make_hook(p)))

    def _make_hook(self, param):
        key = id(param)
        def hook(module, args, output):
            if not module.training:
                return
            if not self._owns.get(key, False):
                return
            if not torch.is_tensor(output) or not output.requires_grad:
                return

            # Save only the detached input. The closure is released once the
            # output-gradient hook has fired for this microbatch.
            x = args[0].detach()

            def output_grad_hook(grad_out):
                z = x.reshape(-1, x.size(-1)).float()        # (N_local, n)
                a = grad_out.detach().reshape(-1, grad_out.size(-1)).float()  # (N_local, m)
                u = row_l2_normalize(a, eps=self._row_eps)   # rsgn(A), rowwise
                n_in = z.size(-1)
                n_out = u.size(-1)
                if key not in self._zz_accum:
                    self._zz_accum[key] = torch.zeros(n_in, n_in, device=z.device, dtype=torch.float32)
                    self._rz_accum[key] = torch.zeros(n_out, n_in, device=z.device, dtype=torch.float32)
                    self._n_accum[key] = 0
                self._zz_accum[key].addmm_(z.T, z)
                self._rz_accum[key].addmm_(u.T, z)
                self._n_accum[key] += z.size(0)

            output.register_hook(output_grad_hook)
        return hook

    @torch.no_grad()
    def step(self):
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
                    n_in = p.shape[-1]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                        state["lookahead_delta"] = torch.zeros_like(p)
                        state["K"] = 1e-3 * torch.eye(n_in, device=p.device, dtype=torch.float32)
                        state["Kinv"] = _ridged_inv(state["K"], group["gamma"], group["eps"])
                        state["step_count"] = 0
                    state["step_count"] += 1

                    key = id(p)
                    N = self._n_accum.get(key, 0)

                    # Since the pullback scale matters, refresh on the first step
                    # whenever data is available, and thereafter every refresh_k.
                    should_refresh = (state["step_count"] == 1) or (state["step_count"] % group["refresh_k"] == 0)
                    if should_refresh and key in self._zz_accum and N > 0:
                        beta = group["beta"]
                        K_batch = self._zz_accum[key] / N
                        if state["step_count"] == 1:
                            # For row-pullback, unlike Muon, the absolute scale
                            # of K^{-1} matters; do not start from a tiny-K EWMA.
                            state["K"].copy_(K_batch)
                        else:
                            state["K"].mul_(beta).add_(K_batch, alpha=1 - beta)
                        state["Kinv"] = _ridged_inv(state["K"], group["gamma"], group["eps"])

                    if key in self._rz_accum and N > 0:
                        # E[rsgn(A)^T z] K^{-1}. This is the positive
                        # gradient-like direction; we subtract lr * update below.
                        rz = self._rz_accum[key] / N
                        direction = (rz @ state["Kinv"]).to(p.dtype)
                    else:
                        # Defensive fallback: if hooks did not fire, use a cruder
                        # row-normalized Newton-Muon-like parameter-space direction.
                        g_precond = (p.grad.float() @ state["Kinv"])
                        direction = row_l2_normalize(g_precond, eps=group["row_eps"]).to(p.dtype)

                    update = momentum_update(direction, state["momentum"], mu=group["mu"])

                    if alpha != 0.0:
                        p.sub_(state["lookahead_delta"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                    if alpha != 0.0:
                        if mode == "deterministic":
                            state["lookahead_delta"].copy_(update).mul_(-alpha * group["lr"])
                        else:  # gaussian; kept in the same parameter-space scaling as Muon/NewtonMuon
                            m, n = p.shape[-2], p.shape[-1]
                            s = group["lr"] * max(1.0, (m / n) ** 0.5) / (m ** 0.5 + n ** 0.5)
                            state["lookahead_delta"].normal_(0.0, s).mul_(alpha)
                        p.add_(state["lookahead_delta"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])

        for k in self._zz_accum:
            self._zz_accum[k].zero_()
            self._rz_accum[k].zero_()
            self._n_accum[k] = 0


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
mbs = 32
train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)


########################################
#       Init & Optim Hyperparams       #
########################################

# CLI args for sweeping. Logged below so the chosen values are recoverable from the logfile.
parser = argparse.ArgumentParser()
parser.add_argument("--optimizer", choices=("muon", "newton_muon", "row_pullback"), default="muon",
                    help="Optimizer for the 2D weights inside transformer blocks. "
                         "AdamW is always used for embed/proj/scalar params.")
parser.add_argument("--lookahead_alpha", type=float, default=0.0,
                    help="Perturbation strength. 0 = standard step. "
                         "alpha=1 shifts the gradient query point by an amount whose operator "
                         "norm equals (or, for gaussian mode, has expected operator norm equal to) "
                         "||x_t - x_{t-1}||_op.")
parser.add_argument("--lookahead_mode", choices=("deterministic", "gaussian"),
                    default="deterministic",
                    help="deterministic: shift along previous update direction (optimism if alpha>0, "
                         "pessimism if alpha<0). gaussian: i.i.d. Gaussian shift, scaled so that at "
                         "alpha=1 the expected spectral norm matches the previous step's spectral norm.")
parser.add_argument("--nm_beta", type=float, default=0.95,
                    help="(newton_muon) EWMA decay on the input Gram matrix K.")
parser.add_argument("--nm_gamma", type=float, default=0.2,
                    help="(newton_muon) Ridge scale: K^{-1} = (K + (gamma*tr(K)/n + eps)*I)^{-1}.")
parser.add_argument("--nm_eps", type=float, default=1e-8,
                    help="(newton_muon) Additive epsilon in the ridge for numerical stability.")
parser.add_argument("--nm_refresh_k", type=int, default=32,
                    help="(newton_muon) Recompute K (and its inverse) every this many steps.")
parser.add_argument("--rpb_beta", type=float, default=0.95,
                    help="(row_pullback) EWMA decay on the input Gram matrix K.")
parser.add_argument("--rpb_gamma", type=float, default=0.2,
                    help="(row_pullback) Ridge scale for K^{-1}.")
parser.add_argument("--rpb_eps", type=float, default=1e-8,
                    help="(row_pullback) Additive epsilon in the ridge for numerical stability.")
parser.add_argument("--rpb_refresh_k", type=int, default=1,
                    help="(row_pullback) Refresh K^{-1} every this many steps. Default 1 because pullback scale matters.")
parser.add_argument("--rpb_mu", type=float, default=0.0,
                    help="(row_pullback) Optional parameter-space momentum after the pullback. 0.0 is the literal derivation; 0.95 is Muon-like.")
parser.add_argument("--rpb_row_eps", type=float, default=1e-12,
                    help="(row_pullback) Epsilon for row-wise output-gradient normalization.")
args = parser.parse_args()
log_hparam(f"cli_args: {vars(args)}")

# build a human-readable tensorboard run name encoding the optimizer + perturbation regime
def _run_tag(optimizer: str, alpha: float, mode: str) -> str:
    opt_prefix = {"muon": "muon", "newton_muon": "nm", "row_pullback": "rpb"}[optimizer]
    if alpha == 0.0:
        regime = "baseline"
    elif mode == "gaussian":
        regime = f"gaussian_a{alpha:g}"
    else:
        regime = (f"optimism_a{alpha:g}" if alpha > 0 else f"pessimism_a{abs(alpha):g}")
    return f"{opt_prefix}_{regime}"
if dist.get_rank() == 0:
    run_tag = _run_tag(args.optimizer, args.lookahead_alpha, args.lookahead_mode)
    writer = SummaryWriter(log_dir=f"runs/{run_tag}_{run_id[:8]}")
    log_hparam(f"tensorboard run dir: runs/{run_tag}_{run_id[:8]}")

# we want to minimize this while still reaching 3.28 val loss
train_steps = 3500

# initialize model parameters
for name, p in model.named_parameters():
    if "proj" in name:
        p.data.zero_()

# create the optimizer(s)
optimizer1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                    dict(params=[model.proj.weight], lr=1/320),
                    dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                   betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=True)
if args.optimizer == "muon":
    optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2],
                      lr=0.025, weight_decay=0.0125,
                      lookahead_alpha=args.lookahead_alpha,
                      lookahead_mode=args.lookahead_mode)
elif args.optimizer == "newton_muon":
    nm_pairs = []
    for m in model.blocks.modules():
        if isinstance(m, nn.Linear) and m.weight.ndim >= 2:
            nm_pairs.append((m.weight, m))
    expected = set(p for p in model.blocks.parameters() if p.ndim >= 2)
    assert set(p for p, _ in nm_pairs) == expected, \
        "NewtonMuon-collected Linear weights must match the Muon param set"
    optimizer2 = NewtonMuon(nm_pairs,
                            lr=0.025, weight_decay=0.0125,
                            beta=args.nm_beta, gamma=args.nm_gamma, eps=args.nm_eps,
                            refresh_k=args.nm_refresh_k,
                            lookahead_alpha=args.lookahead_alpha,
                            lookahead_mode=args.lookahead_mode)
elif args.optimizer == "row_pullback":
    rpb_pairs = []
    for m in model.blocks.modules():
        if isinstance(m, nn.Linear) and m.weight.ndim >= 2:
            rpb_pairs.append((m.weight, m))
    expected = set(p for p in model.blocks.parameters() if p.ndim >= 2)
    assert set(p for p, _ in rpb_pairs) == expected, \
        "RowNormPullback-collected Linear weights must match the Muon param set"
    optimizer2 = RowNormPullback(rpb_pairs,
                                 lr=0.025, weight_decay=0.0125,
                                 mu=args.rpb_mu,
                                 beta=args.rpb_beta, gamma=args.rpb_gamma, eps=args.rpb_eps,
                                 refresh_k=args.rpb_refresh_k, row_eps=args.rpb_row_eps,
                                 lookahead_alpha=args.lookahead_alpha,
                                 lookahead_mode=args.lookahead_mode)
else:
    raise ValueError(f"unknown optimizer: {args.optimizer}")
optimizers = [optimizer1, optimizer2]
assert set(p for opt in optimizers for group in opt.param_groups
           for p in group["params"]) == set(model.parameters())
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

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
    local_loss = torch.zeros((), device="cuda")
    for i in range(len(inputs) // mbs):
        loss = model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
        loss.backward()
        local_loss += loss.detach()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    # gather train-loss + global grad norm before the optimizer mutates grads/params
    if writer is not None:
        dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
        train_loss = local_loss / batch_size
        grad_norm = torch.stack([p.grad.detach().float().pow(2).sum()
                                 for p in model.parameters()]).sum().sqrt()
    # set optimization hyperparameters and take a step
    set_hparams(step)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    if writer is not None:
        writer.add_scalar("loss/train", train_loss.item(), step + 1)
        writer.add_scalar("grad_norm", grad_norm.item(), step + 1)
        writer.add_scalar("lr/muon", optimizer2.param_groups[0]["lr"], step + 1)
        writer.add_scalar("lr/adamw_embed", optimizer1.param_groups[0]["lr"], step + 1)
        writer.add_scalar("lr/adamw_proj", optimizer1.param_groups[1]["lr"], step + 1)
        writer.add_scalar("lr/adamw_scalars", optimizer1.param_groups[2]["lr"], step + 1)
    approx_training_time = training_time + (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
           + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms", console=True, log=False)

if writer is not None:
    writer.close()
dist.destroy_process_group()
