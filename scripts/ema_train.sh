python main_pretrain_ema.py \
    --data_path data \
    --batch_size 256 \
    --epochs 200 \
    --accum_iter 1 \
    --input_size 32 \
    --patch_size 4 \
    --norm_pix_loss \
    --lr 1e-3  