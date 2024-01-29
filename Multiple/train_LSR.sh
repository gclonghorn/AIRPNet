
CUDA_VISIBLE_DEVICES=2 python train_LSR.py\
    -d  /path/to/traning/dataset/ -d_test  /path/to/testing/dataset/\
    --batch-size 9 --val_freq 50  -lr 1e-4 --save  --cuda --exp path_to_experiment\
    --mid 12 --enc 2 2 4 8 --dec 2 2 2 2 --klvl 3 --steps 4 --num_step 12 \
    --cweight1 0 --cweight2 0.01 --sweight1 5  --sweight2 4 --pweight_c 0 --test-patch-size 1024 1024 \
    --hide_checkpoint1 /path/to/1st/stage/LIH \
    --hide_checkpoint2 /path/to/2nd/stage/LIH \
    --hide_checkpoint3 /path/to/GM/ \
    --checkpoint /path/to/LSR/
