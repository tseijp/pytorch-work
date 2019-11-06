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
    if 'vid2pose'==opt.model:
        return Vid2PoseModel(opt)
    if 'pose2vid'==opt.model:
        return Pose2VidModel(opt)
    if 'vid2vid'==opt.model:
        return Vid2Vid(opt)

class BaseModel:
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.Tensor  = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints, opt.name)

    def set_input(self, input):
        self.input = input

    def save(self, label):
        pass

    def load(self, network):
        pass

class Vid2PoseModel(BaseModel):
    def set_model(self):
        model  = opt.model if opt.model else POSE_PATH

class Pose2VidModel(BaseModel):
    def set_model(self):
        model  = opt.model if opt.model else POSE_PATH

class Vid2VidModel(BaseModel):
    def set_model(self):
        model  = opt.model if opt.model else POSE_PATH
