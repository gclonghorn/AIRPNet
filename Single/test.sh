
CUDA_VISIBLE_DEVICES=7 python train_LSR.py\
   -d  /path/to/training/dataset/ -d_test  /path/to/training/dataset/\
    --batch-size 16 --val_freq 50  -lr 1e-4 --save --cuda --exp test \
    --mid 2 --enc 2 2 4 --dec 2 2 2 --klvl 3  --steps 4\
    --num_step 24 --test --test-patch-size 1024 1024 \
    --hide_checkpoint /path/to/LIH/ \
    --checkpoint /path/to/LSR/