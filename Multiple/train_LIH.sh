CUDA_VISIBLE_DEVICES=0 python train_LIH.py -d /path/to/DIV2K_train -d_test /path/to/DIV2K_valid \
 --batch-size 12 -lr1 1e-4 -lr2 1e-4 -lr3 1e-4 --save --cuda --exp /path/to/results \
 --mid 12 --enc 2 2 4 8 --dec 2 2 2 2 --klvl 3 --steps 4 --num_step 12 \ --save_images \
 --num-steps 12 --val-freq 50 --guiding_map True --update_gm True --checkpoint-3 /path/to/GM.tar \
 --checkpoint-1  /path/to/LIH1.tar \
 --checkpoint-2 /path/to/LIH2.tar 

