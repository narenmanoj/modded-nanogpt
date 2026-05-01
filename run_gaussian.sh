for a in 0.25 0.5 1.0 2.0; do
    torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) \
        records/track_3_optimization/train_gpt_simple.py \
        --lookahead_alpha $a --lookahead_mode gaussian
done