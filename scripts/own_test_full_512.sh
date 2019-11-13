############## To test full model #############
##### Using GPUs with 12G memory (not tested)
export CUDA_VISIBLE_DEVICES=0
python own_test_video.py --name everybody_dance_now_temporal --model pose2vid --dataroot ./datasets/ --which_epoch 80 --netG local --ngf 32 --label_nc 0 --no_instance --resize_or_crop none