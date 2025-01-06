CUDA_VISIBLE_DEVICES=0 python train_LSR.py \
    -d /path/to/train/dataset -d_test /path/to/valid/dataset \
    --batch-size 9 --val_freq 50 -lr 1e-5 --save --save_img --cuda --exp /path/to/results \
    --mid 12 --enc 2 2 4 8 --dec 2 2 2 2 --klvl 3 --steps 4 --num_step 12 \
    --cweight1 0 --cweight2 1 --sweight1 6 --sweight2 5 --pweight_c 0 --test-patch-size 1024 1024 \
    --nrate 1 --brate 0 --lrate 0 --degrade_type 1  \
    --hide_checkpoint1 /path/to/hide_checkpoints/net1_checkpoint_checkpoint_best_loss.pth.tar \
    --hide_checkpoint2 /path/to/hide_checkpoints/net2_checkpoint_checkpoint_best_loss.pth.tar \
    --hide_checkpoint3 /path/to/GM.tar \
    --checkpoint /path/to/checkpoints/net_checkpoint_checkpoint_best_loss.pth.tar
