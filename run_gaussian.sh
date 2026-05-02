for a in 0.9 0.8 0.7; do
    torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) \
        records/track_3_optimization/train_gpt_simple.py \
        --lookahead_alpha $a --lookahead_mode gaussian
done