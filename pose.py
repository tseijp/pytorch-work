import os
import cv2
import argparse
import numpy as np
import torch

from lib.network.rtpose_vgg import get_model
from lib.utils.common import draw_humans
from lib.config import cfg, update_config
# my created
from utils.pose_utils import get_outputs, paf_to_pose_cpp

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
update_config(cfg, args)


weight_name = os.path.join(os.getcwd(), 'pose_model.pth')#'/home/tensorboy/Downloads/pose_model.pth'
model = get_model('vgg19')
model.load_state_dict(torch.load(weight_name))
model = torch.nn.DataParallel(model)#.cuda()
model.float()
model.eval()

def arr2pose(arr):
    with torch.no_grad():
        paf, heatmap, im_scale = get_outputs(arr, model,  'rtpose')
    humans = paf_to_pose_cpp(heatmap, paf, cfg) #error
    return humans
    #out = draw_humans(oriImg, humans)
    #cv2.imwrite('result.png',out)
