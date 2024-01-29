
CUDA_VISIBLE_DEVICES=7 python train_LSR.py\
    -d  /path/to/traning/dataset/ -d_test  /path/to/testing/dataset/ \
    --test --test-patch-size 1024 1024\
    --batch-size 9 --val_freq 100  -lr 1e-6 --save  --cuda --exp path_to_experiment\
    --mid 12 --enc 2 2 4 8 --dec 2 2 2 2 --klvl 3 --steps 4 --num_step 12 --sp1 1 --sp2 1 --sp3 1\
    --hide_checkpoint1 /path/to/1st/stage/LIH \
    --hide_checkpoint2 /path/to/2nd/stage/LIH \
    --hide_checkpoint3 /path/to/GM/ \
    --checkpoint /path/to/LSR/