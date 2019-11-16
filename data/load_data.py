# import torch.utils.data as data
#from PIL import Image
#import torchvision.transforms as transforms
#import numpy as np

import torch
from data import aligned_pair_dataset



def load_dataset(opt):#CreateDataset
    dataset = None
    dataset = aligned_pair_dataset.AlignedPairDataset()
    print("dataset [%s] was loaded" % (dataset.name()))
    dataset.initialize(opt) # error
    return dataset

class BaseDataLoader():
    def __init__(self):
        pass
    def initialize(self, opt):
        self.opt = opt
        pass
    def load_data():
        return None

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset    = load_dataset(opt)#CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

def load_data_loader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
