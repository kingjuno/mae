python main_pretrain_bootstrap.py \
    --data_path data \
    --batch_size 256 \
    --no_of_bootstrap 5 \
    --epochs 200 \
    --accum_iter 1 \
    --input_size 32 \
    --patch_size 4 \
    --norm_pix_loss \
    --lr 1e-3  