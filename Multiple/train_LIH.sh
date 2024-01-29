CUDA_VISIBLE_DEVICES=2 python train_LIH.py   -d  /path/to/traning/dataset/ -d_test  /path/to/testing/dataset/ \
--batch-size 12 -lr1 1e-4 -lr2 1e-4 -lr3 1e-4 --save --cuda --exp path_to_experiment \
--num-steps 12 --val-freq 50 --guiding_map --update_gm  \
--checkpoint-1  /path/to/1st/stage/LIH \
--checkpoint-2 /path/to/2nd/stage/LIH \
--checkpoint-3 /path/to/GM/
