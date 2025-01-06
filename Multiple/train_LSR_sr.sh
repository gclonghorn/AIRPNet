
CUDA_VISIBLE_DEVICES=7 python train_LSR.py\
    -d   /mnt/hdd0/gaochao/Dataset/DIV2K_train -d_test  /mnt/hdd0/gaochao/Dataset/DIV2K_valid\
    --batch-size 9 --val_freq 50  -lr 1e-5 --save --save_img --cuda --exp LSR_sr_2\
    --mid 12 --enc 2 2 4 8 --dec 2 2 2 2 --klvl 3 --steps 4 --num_step 12 \
    --cweight1 0 --cweight2 1 --sweight1 8  --sweight2 7 --pweight_c 0 --test-patch-size 1024 1024 \
    --nrate 0 --brate 0 --lrate 1 --degrade_type 3 --test\
    --hide_checkpoint1 /mnt/hdd0/gaochao/PIRNet++/Multiple/pretrained/LIH1.tar \
    --hide_checkpoint2 /mnt/hdd0/gaochao/PIRNet++/Multiple/pretrained/LIH2.tar \
    --hide_checkpoint3 /mnt/hdd0/gaochao/PIRNet++/Multiple/pretrained/GM.tar \
    --checkpoint /mnt/hdd0/gaochao/PIRNet++/Multiple/experiments/LSR_sr/checkpoints/net_checkpoint_checkpoint_best_loss.pth.tar
