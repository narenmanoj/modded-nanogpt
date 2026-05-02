for a in 128 256 512 1024 2048; do
    torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) \
        records/track_3_optimization/train_gpt_simple.py \
        --lookahead_alpha $a --lookahead_mode gaussian
done