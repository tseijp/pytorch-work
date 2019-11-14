import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
    def initialize(self):
        self.parser.add_argument('-f', '--focus',type=float,default=0.33)
        self.parser.add_argument('-a', '--all'  ,type=str  ,nargs='*'  ,default=None,help='all File')
        self.parser.add_argument('-i', '--input',type=str  ,nargs='*'  ,default=None,help='input file')
        self.parser.add_argument('-d', '--display', action='store_true',default=None,)
        self.parser.add_argument('-c', '--chroma' , action='store_true',default=None,)
        self.parser.add_argument('-g', '--is_gpu' , action='store_true',default=None,)
    def parse(self, save=True):
        self.initialize()
        return self.parser.parse_args()

def get_chroma(img_rgb): # return not green position as rgb
    green_min  = np.array([10, 10, 10], np.uint8)
    green_max  = np.array([50, 255, 50], np.uint8)
    green_mask = cv2.inRange(img_rgb, green_min, green_max)
    mask       = cv2.bitwise_not(green_mask)
    mask_rgb   = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return mask_rgb

def get_alpha(img_rgb): # return alpha>0 position as rgb
    img_alp  = img_rgb[:,:,2]
    alp_rgb = cv2.cvtColor(img_alp, cv2.COLOR_GRAY2RGB)
    return alp_rgb

def get_edges(img_rgb):
    img_gry = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_edg = cv2.Canny(img_gry, 10, 100)
    img_edg = cv2.dilate(img_edg, None)
    img_edg = cv2.erode(img_edg, None)
    return cv2.cvtColor(img_edg, cv2.COLOR_GRAY2RGB)

def get_denoise(img_rgb, is_chroma=False): # return denoised image as rgb
    img_msk  = get_chroma(img_rgb) if is_chroma else get_alpha(img_rgb)
    img_edg  = get_edges(img_msk)
    msk_gry  = cv2.cvtColor(img_edg, cv2.COLOR_RGB2GRAY)
    contours = cv2.findContours(msk_gry, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    max_cnt  = max(contours, key=lambda x: cv2.contourArea(x))
    img_dns  = cv2.fillConvexPoly(np.zeros(img_edg.shape), max_cnt[0], (255))
    return img_dns#cv2.cvtColor(img_msk, cv2.COLOR_GRAY2RGB)

def main():
    opt = Options().parse()
    # for setting input
    try:
        path = os.path.join(os.getcwd(), opt.input[0])
        img  = cv2.imread(path, cv2.IMREAD_COLOR)
    except:
        return
    # main process
    images  = [ img,
                get_edges(img),
                get_chroma(img) if opt.chroma else get_alpha(img),
                ]#get_denoise(img, opt.chroma),]
    # display
    if opt.display:
        img_result= cv2.resize(np.concatenate(images, axis=0), dsize=None, fx=opt.focus, fy=opt.focus)
        print(img_result.shape, get_denoise(img, opt.chroma).shape)
        while True:
            # Display the resulting frame
            cv2.imshow('result', img_result)
            cv2.imshow('denoise', cv2.resize(get_denoise(img, opt.chroma), dsize=None, fx=opt.focus, fy=opt.focus))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # When everything is done, release the capture
        cv2.destroyAllWindows()

if __name__=='__main__':
    main()
