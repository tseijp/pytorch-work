import os
#import re
#import sys
import cv2
#import math
#import time
#import scipy
import argparse
#import matplotlib
import numpy as np
#import pylab as plt
import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.autograd import Variable
#from collections import OrderedDict
#from scipy.ndimage.morphology import generate_binary_structure
#from scipy.ndimage.filters import gaussian_filter, maximum_filter

from lib.network.rtpose_vgg import get_model
#from lib.network import im_transform
#from evaluate.coco_eval import get_outputs#, handle_paf_and_heat
#from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender,
from lib.utils.common import draw_humans
from lib.config import cfg, update_config
#from lib.utils.paf_to_pose import paf_to_pose_cpp

# my created
from util.pose_utils import get_outputs, paf_to_pose_cpp

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='./experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='/home/tensorboy/Downloads/pose_model.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)


weight_name = os.path.join(os.getcwd(), 'pose/pose_model.pth')#'/home/tensorboy/Downloads/pose_model.pth'

model = get_model('vgg19')
model.load_state_dict(torch.load(weight_name))
model = torch.nn.DataParallel(model)#.cuda()
model.float()
model.eval()

test_image = 'datasets/test_pose.png'
oriImg     = cv2.imread(test_image) # B,G,R order
print(oriImg.shape)
#shape_dst  = np.min(oriImg.shape[0:2])

# Get results of original image
with torch.no_grad():
    paf, heatmap, im_scale = get_outputs(oriImg, model,  'rtpose')

print(im_scale)
humans = paf_to_pose_cpp(heatmap, paf, cfg) #error
out = draw_humans(oriImg, humans)
cv2.imwrite('result.png',out)

print(human)
