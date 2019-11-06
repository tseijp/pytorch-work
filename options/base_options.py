import os
import argparse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        #self.save = True
        self.initialized = False

    def initialize(self):
        # for setting
        self.parser.add_argument('-w','--epoch'   ,type=str ,default='',help='use epoch pth')
        # for non realtime
        self.parser.add_argument('-i', '--input'  ,type=str ,nargs='*',default=None,help='Input Video Path')
        self.parser.add_argument('-o', '--output' ,type=str ,nargs='*',default=None,help='Output File Path')
        # for realtime
        self.parser.add_argument('-c', '--capture',type=int ,nargs='*',default=None,help='VIdeo Capture Num')
        self.parser.add_argument('-r', '--record' ,action='store_true',help='If set -r,record output as -o name')
        # for me
        #self.parser.add_argument('-d','--debug'   ,action='store_true',help='Set log level to DEBUG')
        #self.parser.add_argument('-e','--error'   ,action='store_true',help='Set log level to ERROR')

        # experiment specifics
        self.parser.add_argument('-n', '--name'  , type=str, default='label2city', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('-m','--model'  , type=str, default='pose2vid', help='which model to use')
        self.parser.add_argument('-g','--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints' , type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--verbose'     , action='store_true', default=False, help='toggles verbose')
        # input/output sizes
        self.parser.add_argument('--loadSize', type=int, default=1024, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
        self.parser.add_argument('--label_nc', type=int, default=35, help='# of input label channels')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        # for setting inputs
        self.parser.add_argument('--dataroot', type=str, default='./datasets/cityscapes/')
        self.parser.add_argument('--resize_or_crop', type=str, default='scale_width', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
        # for generator
        self.parser.add_argument('--netG', type=str, default='global', help='selects model to use for netG')
        self.parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG')
        self.parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')
        # for instance-wise features
        self.parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')
        self.parser.add_argument('--load_features', action='store_true', help='if specified, load precomputed feature maps')

        self.initialized = True

        # for test_video
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--clip_length', type=int, default=500, help='length of generated clip')



    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args(args=[])
        return self.opt
