
CUDA_VISIBLE_DEVICES=7 python train_LSR.py\
   -d  /home/gaochao/Dataset/DIV2K_train/ -d_test  /home/gaochao/Dataset/DIV2K_valid\
    --batch-size 16 --val_freq 50  -lr 1e-4 --save --cuda --exp test \
    --mid 2 --enc 2 2 4 --dec 2 2 2 --klvl 3  --steps 4\
    --cweight 1 --sweight 2 --pweight_c 0.005 --num_step 24 --test --test-patch-size 1024 1024 \
    --hide_checkpoint /home/gaochao/PIRNet++/pretrained/LIH.tar \
    --checkpoint /home/gaochao/PIRNet++/pretrained/LSR.tar