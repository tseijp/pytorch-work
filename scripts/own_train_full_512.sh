############## To train full model #############
##### Using GPUs with 12G memory (not tested)
# Using labels only (0.56s per batch)
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python3 own_train.py --name everybody_dance_now_temporal --model pose2vid --dataroot ./datasets/ --netG local --ngf 32 --num_D 3 --tf_log --niter_fix_global 20 --label_nc 0 --no_instance --no_flow_loss --save_epoch_freq 2 --resize_or_crop none  --continue_train # > train_512_log.txt &

