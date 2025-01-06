CUDA_VISIBLE_DEVICES=0 python train_LIH.py  -d /path/to/train/dataset -d_test /path/to/test/dataset \
--batch-size 16 -lr 1e-4 --save \
--cuda --exp /path/to/results --num-steps 12 --guide-weight 1 --rec-weight 1 --save-images \
--checkpoint /path/to/checkpoint/_checkpoint_best_loss.pth.tar
