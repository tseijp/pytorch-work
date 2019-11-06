### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from data    import load_data
from options import base_options
from models  import load_model

import util
import torch
import numpy as np
from tqdm import tqdm
from imageio import get_writer
from skimage.io import imsave
from collections import OrderedDict

opt = base_options.Options().parse(save=False)
#opt.nThreads = 1   # test code only supports nThreads = 1
#opt.batchSize = 1  # test code only supports batchSize = 1
#opt.serial_batches = True  # no shuffle
#opt.no_flip = True  # no flip
#opt.use_first_frame = False #??

dataset = load_data.load_dataset(opt)
model   = load_model.load_model(opt)
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
