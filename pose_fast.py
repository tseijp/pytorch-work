import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

from skimage.io import imsave, imread
from numpy.random import shuffle
from tensorflow.keras.models import load_model

# from cpu
import argparse
import torch
from pose_comp import cordinates_from_image_file
from lib.config import cfg, update_config
from lib.utils.common import draw_humans
from lib.network.rtpose_vgg import get_model

#from pose import pose_utils
from pose.pose_utils import draw_pose_from_cords , draw_texts
from util.pose_utils import get_outputs, paf_to_pose_cpp, find_peaks
#from lib.pafprocess import pafprocess

parser = argparse.ArgumentParser()
parser.add_argument('--weight', type=str,default='../ckpts/openpose.pth')
parser.add_argument('--is-gpu',action='store_true',default=torch.cuda.is_available())
parser.add_argument('--cfg', default='./experiments/vgg19_368x368_sgd.yaml',type=str)
parser.add_argument('opts',default=None,nargs=argparse.REMAINDER)
opt = parser.parse_args()
cfg.MODEL.NUM_JOINTS    = 19
cfg.MODEL.NUM_KEYPOINTS = 19
update_config(cfg, opt)
opt.is_gpu = True
slow = 0
#if __name__ == "__main__":
if slow:
    # from edn ------------------------------------
    slow_model = load_model('./pose/pose_estimator.h5')
else:
    # from fast -----------------------------------
    weight_name = os.path.join(os.getcwd(), './pose/pose_model.pth')
    model = get_model('vgg19')
    model.load_state_dict(torch.load(weight_name))
    if opt.is_gpu:
        model.cuda()
    model.float()
    model.eval()

if __name__=='__main__':
    img_dir = './datasets/test_B'  # Change this line into where your video frames are stored
    pose_dir = img_dir.replace('test_B', 'test_A')

    #if not os.path.isdir(pose_dir):
    #    os.mkdir(pose_dir)
        #shutil.rmtree(pose_dir)


    comped_list = os.listdir(pose_dir)
    _=[os.remove(os.path.join(pose_dir,  f)) for f in comped_list]
    img_list = os.listdir(img_dir)
    #new_list = [i for i in img_list if not i in comped_list]

    tmp = cv2.imread(os.path.join(img_dir, img_list[0]))
    im_shape = tmp.shape[:-1]

    for item in tqdm(img_list):
        img = cv2.imread(os.path.join(img_dir, item))
        # from edn ------------------------------------
        if slow:
            cord = cordinates_from_image_file(img, model=slow_model)
            #pose_cords.append(cord)
            color,_ = draw_pose_from_cords(cord, im_shape)
            #_=[draw_texts(color, str(i), cord[i]) for i in range(len(cord))
            imsave(os.path.join(pose_dir, item), color)
            break
        # from fast -----------------------------------
        with torch.no_grad():
            paf, heatmap, im_scale = get_outputs(img, model,'rtpose',is_gpu=opt.is_gpu)
        humans = paf_to_pose_cpp(heatmap, paf, cfg) #
        if 1:
            #cord=[[int(bp.y*im_shape[0]),int(bp.x*im_shape[1])] for bp in humans[0].body_parts.values()]
            each = 0
            cord = []
            def mean_bp(c1, c2):
                return [int((c1[0]+c2[0])/2), int((c1[1]+c2[1])/2)]
            if humans:
                for i, bp in humans[0].body_parts.items():
                    if i==each:
                        cord.append([int(bp.y*im_shape[0]),int(bp.x*im_shape[1])])
                    else:
                        cord.append( [None,None] )
                    each += 1
            else:
                pass
            #cord.insert(4, [0,0])
            #cord.insert(5, [0,0])
            #cord[4] = mean_bp(cord[2],cord[3])
            #cord[5] = mean_bp(cord[6],cord[7])

            out,_ = draw_pose_from_cords(cord, im_shape)

            # draw number of joints -----------------------
            #_=[draw_texts(color, str(i), cord[i]) for i in range(len(cord))

            # old draw method 'draw_humans'----------------
            #imageArray = np.zeros((im_shape[0], im_shape[1], 3), np.uint8)
            #out = draw_humans(imageArray, humans)
            #imsave(os.path.join(pose_dir, item), out)#color)

            # save image ----------------------------------
            imsave(os.path.join(pose_dir, item), out)
