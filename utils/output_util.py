from __future__ import absolute_import, division, print_function

import os
import logging
import numpy as np

from skimage import io
from skimage.color import lab2rgb
import cv2
import matplotlib.colors as cl
import matplotlib.pyplot as plt

COS_45 = 1. / np.sqrt(2)
SIN_45 = 1. / np.sqrt(2)
TAG_CHAR = np.array([202021.25], np.float32)
UNKNOWN_FLOW_THRESH = 1e7


def write_depth_png(filename, disp_map):
    io.imsave(filename, (disp_map * 256.0).astype(np.uint16))


def write_flow_png(filename, flow):
    flow = flow * 64.0 + 2 ** 15
    valid = np.ones([flow.shape[0], flow.shape[1], 1])
    flow = np.concatenate([flow, valid], axis=-1).astype(np.uint16)
    flow = np.clip(flow, 0.0, 65535.0).astype(np.uint16)
    cv2.imwrite(filename, flow[..., ::-1])

        

def compute_color_sceneflow(sf):
    """
    scene flow color coding using CIE-LAB space.
    sf: input scene flow, numpy type, size of (h, w, 3)
    """

    # coordinate normalize
    max_sf = np.sqrt(np.sum(np.square(sf), axis=2)).max()
    sf = sf / max_sf

    sf_x = sf[:, :, 0]
    sf_y = sf[:, :, 1]
    sf_z = sf[:, :, 2]
    
    # rotating 45 degree
    # transform X, Y, Z -> Y, X', Z' -> L, a, b 
    sf_x_tform = sf_x * COS_45 + sf_z * SIN_45
    sf_z_tform = -sf_x * SIN_45 + sf_z * COS_45
    sf = np.stack([sf_y, sf_x_tform, sf_z_tform], axis=2) # [-1, 1] cube
    
    # norm vector to lab space: x, y, z -> z, x, y -> l, a, b
    sf[:, :, 0] = sf[:, :, 0] * 50 + 50
    sf[:, :, 1] = sf[:, :, 1] * 127
    sf[:, :, 2] = sf[:, :, 2] * 127
    
    lab_vis = lab2rgb(sf)
    lab_vis = np.uint8(lab_vis * 255)
    lab_vis = np.stack([lab_vis[:, :, 2], lab_vis[:, :, 1], lab_vis[:, :, 0]], axis=2)
    
    return lab_vis

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def flow_to_png_middlebury(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map (h, w, 2)
    :return: optical flow image in middlebury color
    """

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    # maxrad = 80
    
    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def disp_norm_for_vis(disp):
    """
    normalizing disparity for visualization.
    disp: input disparity, np type, size of (h, w)
    """
    disp_sort = np.sort(disp.flatten())
    disp_min = disp_sort[0]
    disp_max = disp_sort[-1]

    return (np.clip((disp - disp_min) / (disp_max - disp_min), 0, 1) * 255).astype(np.uint8)


def save_vis_output(output_tensor, output_path, basename, data_type, save_vis=True, save_output=True):
    
    assert data_type in ['flow', 'disp', 'disp2', 'sf']

    output_np = output_tensor.data.cpu().numpy()
    b_size = output_tensor.size(0)

    file_names = []
    
    # create folder
    for ii in range(b_size):
        file_names.append(os.path.join(output_path, str(basename[ii])))
        directory = os.path.dirname(file_names[-1])
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info("{} has been created.".format(directory))
    
    # save output
    for ii in range(b_size):

        output = output_np[ii, ...].transpose([1, 2, 0]) # b,c,h,w -> h,w,c

        if data_type == 'flow':
            if save_vis:
                io.imsave(file_names[ii] + '_{}.png'.format(data_type), flow_to_png_middlebury(output), check_contrast=False)
            if save_output:
                write_flow_png(file_names[ii] + '_10.png', output) 
        if data_type in ['disp', 'disp2']:
            if save_vis:
                plt.imsave(file_names[ii] + '_{}.jpg'.format(data_type), disp_norm_for_vis(output[:, :, 0]), cmap='plasma')
            if save_output:
                write_depth_png(file_names[ii] + '_10.png', output[:, :, 0])
        if data_type == 'sf':
            if save_vis:
                io.imsave(file_names[ii] + '_{}.png'.format(data_type), compute_color_sceneflow(output), check_contrast=False)





