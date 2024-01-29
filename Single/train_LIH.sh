CUDA_VISIBLE_DEVICES=7 python train_LIH.py  -d  /path/to/training/dataset/ -d_test  /path/to/testing/dataset/   --batch-size 16 -lr 1e-4  --save \
 --cuda --exp path_to_save_checkpoint  --num-steps 24  --guide-weight 1 --rec-weight 2   --checkpoint /path/to/previous/checkpoint ;
