for a in 4.0 8.0 16.0 32.0 64.0; do
    torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) \
        records/track_3_optimization/train_gpt_simple.py \
        --lookahead_alpha $a --lookahead_mode gaussian
done