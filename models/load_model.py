import os
#import numpy as np
import torch
#import torchvision

# from options import base_options

DIRECTORY = os.path.abspath(os.path.dirname(__file__))
CWD_PATH  = os.getcwd()
DATA_PATH = os.path.join(DIRECTORY, 'data')
POSE_PATH = os.path.join(DIRECTORY, 'pose_estimator.pth')
G_PATH = os.path.join(DIRECTORY, 'latest_g.pth')
D_PATH = os.path.join(DIRECTORY, 'latest_d.pth')

def load_model(opt):
    if opt.model == 'pose2vid':
        from .pose2vidHD_model import Pose2VidHDModel, InferenceModel
        if opt.isTrain:
            model = Pose2VidHDModel()
        else:
            model = InferenceModel()
    else:
        from .ui_model import UIModel
        model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model

