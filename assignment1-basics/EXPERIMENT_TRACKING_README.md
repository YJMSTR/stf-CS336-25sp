```

uv run python prepare_data.py --input_file tests/fixtures/tinystories_sample_5M.txt --train_output data/tinystories_sample_5M_train.npy --val_output data/tinystories_sample_5M_val.npy --output_dir data


uv run python train.py \
    --train_data data/tinystories_sample_5M_train.npy \
    --val_data data/tinystories_sample_5M_val.npy \
    --vocab_size 10000 \
    --context_length 512 \
    --d_model 512 \
    --num_layers 4 \
    --num_heads 16 \
    --d_ff 1344 \
    --rope_theta 10000 \
    --batch_size 32 \
    --max_iters 500 --warmup_iters 100 \
    --learning_rate 1e-3 \
    --checkpoint_dir checkpoints/tinystories_bs_32_ctx_512


uv run python generate_text_example.py --checkpoint checkpoints/tinystories_bs_32_ctx_512/checkpoint_final.pt --tokenizer cs336_basics/tinystories_vocab.pkl --merges cs336_basics/tinystories_merges.pkl --prompt 'the little girl'
```