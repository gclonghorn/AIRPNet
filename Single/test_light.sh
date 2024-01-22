
CUDA_VISIBLE_DEVICES=2 python train_LSR.py\
   -d  /path/to/training/dataset/ -d_test  /path/to/testing/dataset/\
    --batch-size 16 --val_freq 50  -lr 1e-6 --save  --cuda --exp test_light \
    --mid 1 --enc 1 1 2 --dec 1 1 1 --klvl 1  --steps 2\
    --num_step 10 --test\
    --hide_checkpoint /path/to/LIH/ \
    --checkpoint /path/to/LSR/