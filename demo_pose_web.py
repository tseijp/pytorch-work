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
from lib.config import cfg, update_config
from lib.utils.common import draw_humans
from lib.network.rtpose_vgg import get_model
#from lib.network import im_transform
#from evaluate.coco_eval import get_outputs, handle_paf_and_heat
#from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender
from lib.pafprocess import pafprocess
# my created
from util.pose_utils import get_outputs, paf_to_pose_cpp, find_peaks

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='./experiments/vgg19_368x368_sgd.yaml', type=str)# my changed
parser.add_argument('--weight', type=str,
                    default='../ckpts/openpose.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
parser.add_argument('--is-gpu', action='store_true', default=False)
args = parser.parse_args()

# update config file
update_config(cfg, args)

weight_name = os.path.join(os.getcwd(), 'pose/pose_model.pth')
#'/home/tensorboy/Downloads/pose_model.pth'                           my changed

model = get_model('vgg19')
model.load_state_dict(torch.load(weight_name))
if args.is_gpu:
    model.cuda()
model.float()
model.eval()

if __name__ == "__main__":
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        # Capture frame-by-frame
        ret, oriImg = video_capture.read()
        #shape_dst = np.min(oriImg.shape[0:2])

        # Get results of original image
        #multiplier = get_multiplier(oriImg)                          my changed

        with torch.no_grad():
            paf, heatmap, im_scale = get_outputs(oriImg, model,'rtpose', is_gpu=args.is_gpu)
            #paf, heatmap = get_outputs(oriImg, model,  'rtpose')     my changed
                #multiplier, oriImg, model,  'rtpose')                my changed

        '''
        heatmap_peaks = np.zeros_like(heatmap)
        for i in range(cfg.MODEL.NUM_KEYPOINTS+1):#19)               my changeed
            print(cfg.MODEL.NUM_KEYPOINTS)
            heatmap_peaks[:,:,i] = find_peaks(cfg.TEST.THRESH_HEATMAP, heatmap[:,:,i])#)my changed
        heatmap_peaks = heatmap_peaks.astype(np.float32)
        heatmap = heatmap.astype(np.float32)
        paf = paf.astype(np.float32)

        #C++ postprocessing
        pafprocess.process_paf(heatmap_peaks, heatmap, paf)

        humans = []
        for human_id in range(pafprocess.get_num_humans()):
            human = Human([])
            is_added = False

            for part_idx in range(18):
                c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
                if c_idx < 0:
                    continue

                is_added = True
                human.body_parts[part_idx] = BodyPart(
                    '%d-%d' % (human_id, part_idx), part_idx,
                    float(pafprocess.get_part_x(c_idx)) / heatmap.shape[1],
                    float(pafprocess.get_part_y(c_idx)) / heatmap.shape[0],
                    pafprocess.get_part_score(c_idx)
                )

            if is_added:
                score = pafprocess.get_score(human_id)
                human.score = score
                humans.append(human)

        out = draw_humans(oriImg, humans)
        '''


        print('scale:',im_scale, '\tfps:',video_capture.get(cv2.CAP_PROP_FPS))
        humans = paf_to_pose_cpp(heatmap, paf, cfg) #error
        out = draw_humans(oriImg, humans)

        # Display the resulting frame
        cv2.imshow('Video', out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


'''
def draw_humans(npimg, humans, imgcopy=False):
    if imgcopy:
        npimg = np.copy(npimg)
    image_h, image_w = npimg.shape[:2]
    centers = {}
    for human in humans:
        # draw point
        for i in range(CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue

            body_part = human.body_parts[i]
            center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
            centers[i] = center
            cv2.circle(npimg, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)

        # draw line
        for pair_order, pair in enumerate(CocoPairsRender):
            if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                continue

            # npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)
            cv2.line(npimg, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)

    return npimg
'''
