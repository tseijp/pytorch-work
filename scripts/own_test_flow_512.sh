export CUDA_VISIBLE_DEVICES=0,1,2,3
python own_test_video.py --name everybody_dance_now_temporal --model pose2vid --dataroot ./datasets/ --which_epoch latest --netG local --ngf 32 --label_nc 0 --no_instance --resize_or_crop scale_width --loadSize 512
