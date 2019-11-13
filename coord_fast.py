import os
import numpy as np

from tensorflow.keras.models import load_model
#import skimage.transform as st
#import pandas as pd
from tqdm import tqdm
from numpy.random import shuffle
#from skimage.transform import resize
#from scipy.ndimage import gaussian_filter
from skimage.io import imsave, imread

# from cpu
import argparse
import torch
from lib.config import cfg, update_config
from lib.utils.common import draw_humans
from lib.network.rtpose_vgg import get_model

from pose import pose_utils
from coord_comp import compute_cordinates,cordinates_from_image_file,strip_frames,check_validity
from pose.pose_utils import draw_pose_from_cords
from utils.pose_utils import get_outputs, paf_to_pose_cpp, find_peaks
#from lib.pafprocess import pafprocess

parser = argparse.ArgumentParser()
parser.add_argument('--weight', type=str,default='../ckpts/openpose.pth')
parser.add_argument('--is-gpu', action='store_true', default=False)
parser.add_argument('--cfg', default='./experiments/vgg19_368x368_sgd.yaml',type=str)
parser.add_argument('opts',default=None,nargs=argparse.REMAINDER)
opt = parser.parse_args()
update_config(cfg, opt)


if __name__ == "__main__":
    img_dir = './datasets/train_B'  # Change this line into where your video frames are stored
    pose_dir = img_dir.replace('train_B', 'train_A')
    pose_npy_name = img_dir.replace('train_B', 'poses.npy')
    if not os.path.isdir(pose_dir):
        os.mkdir(pose_dir)
    img_dir = img_dir.replace('train_B', 'test_pose')# now changed
    img_list = os.listdir(img_dir)
    tmp = imread(os.path.join(img_dir, img_list[0]))
    im_shape = tmp.shape[:-1]

    # from edn ------------------------------------
    #slow_model = load_model('./pose/pose_estimator.h5')
    # from fast -----------------------------------
    weight_name = os.path.join(os.getcwd(), './pose/pose_model.pth')
    model = get_model('vgg19')
    model.load_state_dict(torch.load(weight_name))
    if opt.is_gpu:
        model.cuda()
    model.float()
    model.eval()

    pose_cords = []
    for item in tqdm(img_list):
        img = imread(os.path.join(img_dir, item))
        # from fast -----------------------------------
        with torch.no_grad():
            paf, heatmap, im_scale = get_outputs(img, model,'rtpose',is_gpu=opt.is_gpu)
        humans = paf_to_pose_cpp(heatmap, paf, cfg) #
        out = draw_humans(img, humans)
        imsave('result2.png', out)
        # from edn ------------------------------------
        '''
        cord = cordinates_from_image_file(img, model=slow_model)
        pose_cords.append(cord)
        '''
        color,_ = draw_pose_from_cords(humans, im_shape)
        #imsave(os.path.join(pose_dir, item), color)
        imsave('result1.png', color)


        with open('./log.txt', mode='w') as f:
            for i in range(4):f.write('\n--paf--\t')
            f.write('%s'%paf)
            for i in range(4):f.write('\n--heatmap--\t')
            f.write('%s'%heatmap)
            for i in range(4):f.write('\n--humans--\t')
            f.write('%s'%humans)
            for i in range(4):f.write('\n--cord--\t')
            f.write('%s'%cord)

    np.save(pose_npy_name, np.array(pose_cords, dtype=np.int))
