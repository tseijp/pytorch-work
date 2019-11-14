#!/usr/bin/env python
# -*- coding: utf-8 -*-
### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

import os
from collections import OrderedDict
from options.test_options import TestOptions
from data.load_data import load_dataset
from models.load_model import load_model
import util.util as util
import torch
from imageio import get_writer
import numpy as np
from tqdm import tqdm
import time

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

dataset = load_dataset(opt)

# test
model = load_model(opt)
if opt.verbose:
    print(model)


def my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    
# test whole video sequence
# 20181009: do we use first frame as input?
time_now = str(time.gmtime().tm_year) + str(time.gmtime().tm_mon) + str(time.gmtime().tm_mday) + str(time.gmtime().tm_hour) + str(time.gmtime().tm_min)
data = dataset[0]
if opt.use_first_frame:
    prev_frame = data['image']
    start_from = 1
    from skimage.io import imsave
    imsave('results/ref.png', util.tensor2im(prev_frame))
    generated = [util.tensor2im(prev_frame)]
else:
    prev_frame = torch.zeros_like(data['image'])
    start_from = 0
    generated = []

from skimage.io import imsave
for i in tqdm(range(start_from, dataset.clip_length)):
    label = data['label'][i:i+1]
    #print(label.shape)
    inst = None if opt.no_instance else data['inst'][i:i+1]

    cur_frame = model.inference(label, inst, torch.unsqueeze(prev_frame, dim=0))
    prev_frame = cur_frame.data[0]
    new_dir_path_recursive = '.datasets/own_dance_test/test_sync'+str(time_now)
    my_makedirs(new_dir_path_recursive)
    imsave(new_dir_path_recursive+'/{:05d}.png'.format(i), util.tensor2im(prev_frame))
    generated.append(util.tensor2im(prev_frame))
'''
result_dir = os.path.join(opt.results_dir, opt.name, opt.which_epoch)
if not os.path.isdir(result_dir):
  os.makedirs(result_dir, exist_ok=True)

with get_writer(os.path.join(result_dir, 'test_clip_ref.avi' if opt.use_first_frame else 'test_clip.avi'), fps=25) as writer:
  for im in generated:
      writer.append_data(im)
writer.close()
'''
