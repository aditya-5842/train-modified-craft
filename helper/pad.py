import cv2
import numpy as np


def pad_img(img, width, height):
    """
    """
    width = width#224
    height = height#96
    pad_type = ''
    h, w, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if w<=width and h<=height:
        if w!=width:
            temp = np.ones((h, width-w, 3), np.uint8)*255
            img = np.hstack([img, temp])
            pad_type += 'wLess'
        if h!=height:
            temp = np.ones((height-h, width, 3), np.uint8)*255
            img = np.vstack([img, temp])
            pad_type += '_hLess'
        if w==width and h==height:
            pad_type = 'wEq_hEq'
        elif w<width and h==height:
            pad_type = 'wLess_hEq'
        elif w==width and h<height:
            pad_type = 'wEq_hLess'
    elif w>width and h<=height:
        img = cv2.resize(img, (width, h))
        if h!=height:
            temp = np.ones((height-h, width, 3), np.uint8)*255
            img = np.vstack([img, temp])
            pad_type = 'wGr_hLess'
        else:
            pad_type = 'wGr_hEq'
    elif w<=width and h>height:
        img = cv2.resize(img, (w, height))
        if w!=width:
            temp = np.ones((height, width-w, 3), np.uint8)*255
            img = np.hstack([img, temp])
            pad_type = 'wLess_hGr'
        else:
            pad_type = 'wEq_hGr'
    elif w>width and h>height:
        img = cv2.resize(img, (width, height))
        pad_type = 'wGr_hGr'
    return img, pad_type


def decode_boxes(padded_img, org_img, boxes, pad_type):
    """
    """
    h, w, _ = org_img.shape
    height, width, _ = padded_img.shape
    # case: 1 (pad_type == 'wEq_hEq')
    if pad_type=='wEq_hEq':
        return boxes

    # case: 2 (pad_type == 'wLess_hLess), it may need correction
    if pad_type=='wLess_hLess':
        return boxes

    # case: 3 (pad_type == 'wGr_hGr)
    if pad_type=='wGr_hGr':
        ratio_mat = np.array([w/width, h/height], np.float32)
        boxes = boxes*ratio_mat
        return boxes

    # case: 4 (pad_type == 'wGr_hLess)
    if pad_type=='wGr_hLess':
        height = h
        ratio_mat = np.array([w/width, h/height], np.float32)
        boxes = boxes*ratio_mat
        return boxes

    # case: 5 (pad_type == 'wLess_hGr)
    if pad_type=='wLess_hGr':
        width = w
        ratio_mat = np.array([w/width, h/height], np.float32)
        boxes = boxes*ratio_mat
        return boxes

    # case: 6 (pad_type == 'wLess_hEq)
    if pad_type=='wLess_hEq':
        return boxes

    # case: 7 (pad_type == 'wGr_hEq)
    if pad_type=='wGr_hEq':
        height = h
        ratio_mat = np.array([w/width, h/height], np.float32)
        boxes = boxes*ratio_mat
        return boxes

    # case: 8 (pad_type == 'wEq_hLess)
    if pad_type=='wEq_hLess':
        return boxes

    # case: 9 (pad_type == 'wEq_hGr)
    if pad_type=='wEq_hGr':
        width = w
        ratio_mat = np.array([w/width, h/height], np.float32)
        boxes = boxes*ratio_mat
        return boxes

