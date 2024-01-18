CUDA_VISIBLE_DEVICES=7 python train_LIH.py  -d  /home/gaochao/Dataset/DIV2K_train/ -d_test  /home/gaochao/Dataset/DIV2K_valid/   --batch-size 16 -lr 1e-4  --save \
 --cuda --exp test  --num-steps 24  \
 --guide-weight 1 --rec-weight 2   --checkpoint /home/gaochao/PIRNet++/pretrained/LIH.tar ;
