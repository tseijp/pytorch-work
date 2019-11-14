
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import torch.utils.data as data
#from PIL import Image
#import torchvision.transforms as transforms
#import numpy as np

from data import aligned_pair_dataset

def load_dataset(opt):#CreateDataset
    dataset = None
    dataset = aligned_pair_dataset.AlignedPairDataset()
    print("dataset [%s] was loaded" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

# do not use ???
'''
class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset    = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
'''
