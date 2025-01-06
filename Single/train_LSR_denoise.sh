CUDA_VISIBLE_DEVICES=0 python train_LSR.py \
   -d /path/to/train/dataset -d_test /path/to/valid/dataset \
   --batch-size 16 --val_freq 50 -lr 1e-4 --save --cuda --exp /path/to/results --nafwidth 32 \
   --mid 2 --enc 2 2 4 --dec 2 2 2 --klvl 3 --steps 4 --save_img \
   --cweight 1 --sweight 7 --pweight_c 0.005 --num_step 12 --test-patch-size 1024 1024 \
   --nrate 1 --brate 0 --lrate 0 --degrade_type 1  \
   --hide_checkpoint /path/to/hide/checkpoint/_checkpoint_best_loss.pth.tar \
   --checkpoint /path/to/checkpoint/_checkpoint_best_loss.pth.tar
