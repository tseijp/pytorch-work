import os
import shutil
import numpy as np
from tqdm import tqdm

from skimage.io import imsave, imread
from numpy.random import shuffle
from tensorflow.keras.models import load_model

# from cpu
import argparse
import torch
from lib.config import cfg, update_config
from lib.utils.common import draw_humans
from lib.network.rtpose_vgg import get_model

#from pose import pose_utils
from pose.pose_utils import draw_pose_from_cords
from util.pose_utils import get_outputs, paf_to_pose_cpp, find_peaks
#from lib.pafprocess import pafprocess

parser = argparse.ArgumentParser()
parser.add_argument('--weight', type=str,default='../ckpts/openpose.pth')
parser.add_argument('--is-gpu',action='store_true',default=torch.cuda.is_available())
parser.add_argument('--cfg', default='./experiments/vgg19_368x368_sgd.yaml',type=str)
parser.add_argument('opts',default=None,nargs=argparse.REMAINDER)
opt = parser.parse_args()
update_config(cfg, opt)

if __name__ == "__main__":
    # from edn ------------------------------------
    slow_model = load_model('./pose/pose_estimator.h5')
    # from fast -----------------------------------
    weight_name = os.path.join(os.getcwd(), './pose/pose_model.pth')
    model = get_model('vgg19')
    model.load_state_dict(torch.load(weight_name))
    if opt.is_gpu:
        model.cuda()
    model.float()
    model.eval()

    img_dir = './datasets/test_B'  # Change this line into where your video frames are stored
    pose_dir = img_dir.replace('test_B', 'test_A')

    if os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    os.mkdir(pose_dir)

    comped_list = os.listdir(pose_dir)
    img_list = os.listdir(img_dir)
    new_list = [i for i in img_list if not i in comped_list]

    tmp = imread(os.path.join(img_dir, img_list[0]))
    im_shape = tmp.shape[:-1]

    for item in tqdm(new_list):
        img = imread(os.path.join(img_dir, item))
        # from fast -----------------------------------
        with torch.no_grad():
            paf, heatmap, im_scale = get_outputs(img, model,'rtpose',is_gpu=opt.is_gpu)
        humans = paf_to_pose_cpp(heatmap, paf, cfg) #
        cord=[]
        for bp in humans[0].body_parts.values():
            bp_x =int( bp.x*im_shape[1] )
            bp_y =int( bp.y*im_shape[0] )
            cord.append([bp_x, bp_y])
        print(cord)
        #color,_= draw_pose_from_cords(humans, im_shape)
        #imsave(os.path.join(pose_dir, item), color)
        # from edn ------------------------------------
        cord = cordinates_from_image_file(img, model=slow_model)
        print(cord)
        #imsave(os.path.join(pose_dir, item), color)
        #imsave('result1.png', color)
