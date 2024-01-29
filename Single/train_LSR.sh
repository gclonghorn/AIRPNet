
CUDA_VISIBLE_DEVICES=7 python train_LSR.py\
   -d  /path/to/training/dataset/ -d_test  /path/to/testing/dataset/\
    --batch-size 16 --val_freq 50  -lr 1e-4 --save --cuda --exp path_to_save_checkpoint \
    --mid 2 --enc 2 2 4 --dec 2 2 2 --klvl 3 \
    --cweight 1 --sweight 2 --pweight_c 0.005 --num_step 24  --test-patch-size 1024 1024 \
    --hide_checkpoint /path/to/LIH \
    --checkpoint /path/to/LSR