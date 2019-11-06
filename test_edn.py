import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from imageio import get_writer
from collections import OrderedDict

from utils   import edn_utils as util
from models  import load_model
from data    import load_data
from pose    import pose_estimator
from options import base_options

opt     = base_options.Options().parse()
opt.use_first_frame = False
opt.no_instance = True or False

dataset    = data_loader.load_dataset(opt)
model      = model_loader.load_model(opt)

data       = dataset[0] #index 0 is teonsor of A_tensor
prev_frame = torch.zeros_like(data['image'])
generated  = []

def no_loop():
    for i in tqdm(0, datasets.clip_length):
        label = data['label'][i:i+1]# label is A_tensor
        inst = None if opt.no_instance else data['inst'][i:i+1]

        cur_frame  = model.inference(label, inst, torch.unsqueeze(prev_frame, dim=0))
        prev_frame = cur_frame.data[0]

        imsave('./datasets/cardio_dance_test/test_sync/{:05d}.png'.format(i), util.tensor2im(prev_frame))
        generated.append(util.tensor2im(prev_frame))




if __name__=='__main__':
    if opt.capture:
        cap = cv2.VideoCapture(opt.capture)  # capture from file
    else:
        cap = cv2.VideoCapture(0)   # capture from camera
        if not cap.isOpened():
            raise ImportError("Couldn't open video file or webcam.")

    frame = 0
    while True:
        ret, orig_image = cap.read()
        if orig_image is None:
            continue

        # process ------------------------------------------
        input   = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        (height, width) = image.shape
        timer.start()
        # model
        output  = model(input)

        interval= timer.end()
        # --------------------------------------------------

        # display
        print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
        fps = "FPS:" + str(int(1/interval))
        cv2.imshow("SSD Result", output)
        # imsave
        if opt.record:
            cv2.imsave('%s'%frame.zfill(5)+'.png', output)
        # stop
        if cv2.waitKey(1) & 0xFF == ord('q') or frame>99999:
            break

        frame += 1
