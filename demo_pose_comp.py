import os
import numpy as np
import shutil

from tensorflow.keras.models import load_model
import skimage.transform as st
import pandas as pd
from tqdm import tqdm
from numpy.random import shuffle
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from skimage.io import imsave, imread

# made
from options.demo_options import DemoPoseOption
from pose import pose_utils
from pose.pose_utils import draw_pose_from_cords
from pose_comp import cordinates_from_image_file

#opt = DemoPoseOption().parse(save=False)

if __name__ == "__main__":
    img_dir = './datasets/test_B'  # Change this line into where your video frames are stored
    pose_dir = img_dir.replace('test_B', 'test_A')
    pose_npy_name = img_dir.replace('test_B', 'poses.npy')
    if os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    os.mkdir(pose_dir)
    #img_dir = img_dir.replace('test_B', 'test_pose')

    model = load_model('./pose/pose_estimator.h5')
    comped_list = os.listdir(pose_dir)
    img_list = os.listdir(img_dir)
    new_list = [i for i in img_list if not i in comped_list]

    tmp = imread(os.path.join(img_dir, img_list[0]))
    im_shape = tmp.shape[:-1]

    for item in tqdm(new_list): # my changed
        img = imread(os.path.join(img_dir, item))
        cord = cordinates_from_image_file(img, model=model)
        #pose_cords.append(cord)
        color,_ = draw_pose_from_cords(cord, im_shape)
        imsave(os.path.join(pose_dir, item), color)
