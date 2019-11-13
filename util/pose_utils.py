import cv2
import torch
import numpy as np
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure

from lib.config import cfg, update_config
from lib.network import im_transform
from lib.pafprocess import pafprocess
from lib.utils.common import Human, BodyPart
from lib.datasets.preprocessing import rtpose_preprocess#(inception_preprocess,
                                        #rtpose_preprocess,
                                        #ssd_preprocess, vgg_preprocess)

def get_outputs(img, model, preprocess, is_gpu=False):
    inp_size = cfg.DATASET.IMAGE_SIZE
    # padding
    im_croped, im_scale, real_shape = im_transform.crop_with_factor(
        img, inp_size, factor=cfg.MODEL.DOWNSAMPLE, is_ceil=True)

    if preprocess   == 'rtpose':      im_data = rtpose_preprocess(im_croped)
    #elif preprocess == 'vgg':       im_data = vgg_preprocess(im_croped)
    #elif preprocess == 'inception': im_data = inception_preprocess(im_croped)
    #elif preprocess == 'ssd':       im_data = ssd_preprocess(im_croped)

    batch_images= np.expand_dims(im_data, 0)
    # several scales as a batch
    batch_var = torch.from_numpy(batch_images).float()#.cuda().float()
    if is_gpu:
        batch_var = batch_var.cuda().float()

    predicted_outputs, _ = model(batch_var)
    output1, output2 = predicted_outputs[-2], predicted_outputs[-1]
    heatmap = output2.cpu().data.numpy().transpose(0, 2, 3, 1)[0]
    paf = output1.cpu().data.numpy().transpose(0, 2, 3, 1)[0]

    return paf, heatmap, im_scale

def find_peaks(param, img):

    peaks_binary = (maximum_filter(img, footprint=generate_binary_structure(
        2, 1)) == img) * (img > param)
    # Note reverse ([::-1]): we return [[x y], [x y]...] instead of [[y x], [y
    # x]...]
    return np.array(np.nonzero(peaks_binary)[::-1]).T

def compute_resized_coords(coords, resizeFactor):
    return (np.array(coords, dtype=float) + 0.5) * resizeFactor - 0.5

def NMS(heatmaps, upsampFactor=1., bool_refine_center=True, bool_gaussian_filt=False, config=None):

    joint_list_per_joint_type = []
    cnt_total_joints = 0
    win_size = 2

    for joint in range(config.MODEL.NUM_KEYPOINTS):
        map_orig = heatmaps[:, :, joint]
        peak_coords = find_peaks(config.TEST.THRESH_HEATMAP, map_orig)
        peaks = np.zeros((len(peak_coords), 4))
        for i, peak in enumerate(peak_coords):
            if bool_refine_center:
                x_min, y_min = np.maximum(0, peak - win_size)
                x_max, y_max = np.minimum(
                    np.array(map_orig.T.shape) - 1, peak + win_size)

                # Take a small patch around each peak and only upsample that
                # tiny region
                patch = map_orig[y_min:y_max + 1, x_min:x_max + 1]
                map_upsamp = cv2.resize(
                    patch, None, fx=upsampFactor, fy=upsampFactor, interpolation=cv2.INTER_CUBIC)

                # Gaussian filtering takes an average of 0.8ms/peak (and there might be
                # more than one peak per joint!) -> For now, skip it (it's
                # accurate enough)
                map_upsamp = gaussian_filter(
                    map_upsamp, sigma=3) if bool_gaussian_filt else map_upsamp

                # Obtain the coordinates of the maximum value in the patch
                location_of_max = np.unravel_index(
                    map_upsamp.argmax(), map_upsamp.shape)
                # Remember that peaks indicates [x,y] -> need to reverse it for
                # [y,x]
                location_of_patch_center = compute_resized_coords(
                    peak[::-1] - [y_min, x_min], upsampFactor)
                # Calculate the offset wrt to the patch center where the actual
                # maximum is
                refined_center = (location_of_max - location_of_patch_center)
                peak_score = map_upsamp[location_of_max]
            else:
                refined_center = [0, 0]
                # Flip peak coordinates since they are [x,y] instead of [y,x]
                peak_score = map_orig[tuple(peak[::-1])]
            peaks[i, :] = tuple(
                x for x in compute_resized_coords(peak_coords[i], upsampFactor) + refined_center[::-1]) + (
                              peak_score, cnt_total_joints)
            cnt_total_joints += 1
        joint_list_per_joint_type.append(peaks)

    return joint_list_per_joint_type

def paf_to_pose_cpp(heatmaps, pafs, config):
    humans = []
    joint_list_per_joint_type = NMS(heatmaps, upsampFactor=config.MODEL.DOWNSAMPLE, config=config)

    joint_list = np.array(
        [tuple(peak) + (joint_type,) for joint_type, joint_peaks in enumerate(joint_list_per_joint_type) for peak in
         joint_peaks]).astype(np.float32)

    if joint_list.shape[0] > 0:
        joint_list = np.expand_dims(joint_list, 0)
        paf_upsamp = cv2.resize(
            pafs, None, fx=config.MODEL.DOWNSAMPLE, fy=config.MODEL.DOWNSAMPLE, interpolation=cv2.INTER_NEAREST)
        heatmap_upsamp = cv2.resize(
            heatmaps, None, fx=config.MODEL.DOWNSAMPLE, fy=config.MODEL.DOWNSAMPLE, interpolation=cv2.INTER_NEAREST)
        pafprocess.process_paf(joint_list, heatmap_upsamp, paf_upsamp)
        for human_id in range(pafprocess.get_num_humans()):
            human = Human([])
            is_added = False
            for part_idx in range(config.MODEL.NUM_KEYPOINTS):
                c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
                if c_idx < 0:
                    continue
                is_added = True
                human.body_parts[part_idx] = BodyPart(
                    '%d-%d' % (human_id, part_idx), part_idx,
                    float(pafprocess.get_part_x(c_idx)) / heatmap_upsamp.shape[1],
                    float(pafprocess.get_part_y(c_idx)) / heatmap_upsamp.shape[0],
                    pafprocess.get_part_score(c_idx)
                )
            if is_added:
                score = pafprocess.get_score(human_id)
                human.score = score
                humans.append(human)

    return humans
