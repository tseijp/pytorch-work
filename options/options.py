import os
import argparse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        #self.save = True
        self.initialized = False

    def initialize(self):
        # for setting input
        self.parser.add_argument('-a', '--all'   ,type=str ,nargs='*',default=None,help='Convert all file of dir')
        self.parser.add_argument('-i', '--input' ,type=str ,nargs='*',default=None,help='Input file to convert')
        self.parser.add_argiment('--is_gpu' , type='store_true')
    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt
