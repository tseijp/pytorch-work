### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from options.base_options import Options
#from data.custom_dataset_data_loader import CreateDataset
from data.load_data import load_dataset
from models.load_model import load_model

from utils import pose_utils as util
import torch
import numpy as np
from tqdm import tqdm
from imageio import get_writer
from skimage.io import imsave

opt = Options().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.use_first_frame = False #??

dataset = load_dataset(opt)#CreateDataset(opt)
model   = load_model(opt)#create_model(opt)
data    = dataset[0]

prev_frame = torch.zeros_like(data['image'])
start_from = 0
generated  = []

for i in tqdm(range(start_from, dataset.clip_length)):
    label = data['label'][i:i+1]
    inst = None if opt.no_instance else data['inst'][i:i+1]

    cur_frame = model.inference(label, inst, torch.unsqueeze(prev_frame, dim=0))
    prev_frame = cur_frame.data[0]

    imsave('./datasets/cardio_dance_test/test_sync/{:05d}.png'.format(i), util.tensor2im(prev_frame))
    generated.append(util.tensor2im(prev_frame))
