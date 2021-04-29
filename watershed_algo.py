import os
import glob
import json
import imgaug
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework.dtypes import as_dtype
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from sklearn.model_selection import train_test_split
from helper.dataset import get_image_data_generator, DataGenerator
from helper.build_model import build_keras_model, load_keras_model_vgg16, build_effiecient_basenetmodel

################################ load data (csv file) and data slplit ######################
df = pd.read_csv("./labeled.csv")
df['boxes'] = df['boxes'].apply(lambda x: json.loads(x))
labeles = []

for _, row in df.iterrows():
    boxes = []
    for box in row['boxes']:
        boxes.append([(np.array(box, np.float), 'a')])
    labeles.append((row['filename'], boxes, 1))

# data split
train, validation = train_test_split(labeles, train_size=0.8, random_state=42)
############################################################################################

def get_boxes(y_pred, detection_threshold=0.7, text_threshold=0.1, size_threshold=10):
    box_groups = []
    for textmap in y_pred:
        # Prepare data
        img_h, img_w = textmap.shape

        _, text_score = cv2.threshold(textmap, thresh=text_threshold, maxval=1, type=cv2.THRESH_BINARY)
        n_components, labels, stats, _ = cv2.connectedComponentsWithStats(np.clip(text_score, 0, 1).astype('uint8'), connectivity=4)
        boxes = []
        for component_id in range(1, n_components):
            # Filter by size
            size = stats[component_id, cv2.CC_STAT_AREA]

            if size < size_threshold:
                continue

            # If the maximum value within this connected component is less than
            # text threshold, we skip it.
            if np.max(textmap[labels == component_id]) < detection_threshold:
                continue

            # Make segmentation map. It is 255 where we find text, 0 otherwise.
            segmap = np.zeros_like(textmap)
            segmap[labels == component_id] = 255
            x, y, w, h = [stats[component_id, key] for key in [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]

            # Expand the elements of the segmentation map
            niter = int(np.sqrt(size * min(w, h) / (w * h)) * 2)
            sx, sy = max(x - niter, 0), max(y - niter, 0)
            ex, ey = min(x + w + niter + 1, img_w), min(y + h + niter + 1, img_h)
            segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex],cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter)))

            # Make rotated box from contour
            contours = cv2.findContours(segmap.astype('uint8'), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)[-2]
            contour = contours[0]
            box = cv2.boxPoints(cv2.minAreaRect(contour))

            # Check to see if we have a diamond
            w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:
                l, r = contour[:, 0, 0].min(), contour[:, 0, 0].max()
                t, b = contour[:, 0, 1].min(), contour[:, 0, 1].max()
                box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
            else:
                # Make clock-wise order
                box = np.array(np.roll(box, 4 - box.sum(axis=1).argmin(), 0))
            boxes.append(2 * box)
        box_groups.append(np.array(boxes))
    return box_groups

def remove_border(Z):
    # top row
    mask = Z[1,:]>1
    Z[0,:][~mask] = 1
    # right
    mask = Z[:,-2]>1
    Z[:,-1][~mask] = 1
    # bottom
    mask = Z[-2,:]>1
    Z[-1,:][~mask] = 1
    #left
    mask = Z[:,1]>1
    Z[:,0][~mask] = 1
    return Z

def get_boxes_after_watershed(Z, increament=0.25):
    boxes = []
    H, W = Z.shape
    for label in range(2, Z.max()+1):
        A = Z.copy()
        A[A==label] = 255
        A[A!=255] = 0
        A = A.astype(np.uint8)
        contours, hierarchy = cv2.findContours(A,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)>0:
            if len(contours)==1:
                x, y, w, h = cv2.boundingRect(contours[0])
                x_center, y_center = x+w/2, y+h/2

                # increase width and height
                w = w*(1+increament)
                h = h*(1+increament)
                
                x, y = max(0,int(x_center-w/2)), max(0, int(y_center-h/2))

                box = [[x,y], [min(x+w, W), y], [min(x+w, W), min(y+h, H)], [x, min(y+h, H)]]
                boxes.append(box)
            else:
                # pick the countour with maximum area
                contour_areas = [cv2.contourArea(cnt) for cnt in contours]
                idx = contour_areas.index(max(contour_areas))
                x,y,w,h = cv2.boundingRect(contours[idx])
                box = [[x,y], [x+w, y], [x+w, y+h], [x, y+h]]
                boxes.append(box)
    boxes = np.array(boxes, np.int)*2
    return boxes


################################ image augmentor and data genrator #########################
augmenter = imgaug.augmenters.Sequential([
    imgaug.augmenters.GaussianBlur(sigma=(0, 3)), #Gaussian blur with mean=0 and std = 3
    imgaug.augmenters.Affine(scale=(1, 1), rotate = (-5, 5)),
    imgaug.augmenters.Multiply((0.8, 1.2), per_channel=0.2),
    imgaug.augmenters.GammaContrast(gamma=(0.25, 3.0))]
)

generator_kwargs = {'width': 224, 'height': 224}

train_img_gen = get_image_data_generator(labels=train, augmenter=augmenter, **generator_kwargs)
valid_img_gen = get_image_data_generator(labels=validation, **generator_kwargs)

train_gen = DataGenerator(image_generator=train_img_gen)
valid_gen = DataGenerator(image_generator=valid_img_gen)

x_train, y_train = train_gen.get_item()
x_val, y_val = valid_gen.get_item()


def craft_decode1(y_train):
    box_groups = get_boxes(y_train, text_threshold=0.18)
    return box_groups

def craft_decode2(y, img):
    bg = np.where(y>0.01, 1, 0)*255
    bg = bg.astype(np.uint8)

    # dilate bg
    kernel = np.ones((2,2),np.uint8)
    bg = cv2.dilate(bg,kernel,iterations=1)

    fg = np.where(y>0.4, 1, 0)*255
    fg = fg.astype(np.uint8)

    unknown = cv2.subtract(bg,fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    Z = cv2.watershed(img, markers.copy())
    Z = remove_border(Z)
    boxes = get_boxes_after_watershed(Z)

    return boxes

def craft_decode3(y, thres=0.4, increament_w=0.8, increament_h=1.0):
    H, W = y.shape
    fg = np.where(y>=thres, 1, 0)*255
    fg = fg.astype(np.uint8)

    # apply opening to remove the noisy pixels
    kernel = np.ones((2,2),np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(fg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_center, y_center = x+w/2-1, y+h/2-2

        # increase width and height
        w = w*(1+increament_w)
        h = h*(1+increament_h)
        if w*h<5:
            continue
        x, y = max(0,math.floor(x_center-w/2)), max(0, math.floor(y_center-h/2))
        box = [
            [x,y], 
            [min(math.ceil(x+w), W), y], 
            [min(math.ceil(x+w), W), min(math.ceil(y+h), H)],
            [x, min(math.ceil(y+h), H)]
        ]
        boxes.append(box)

    boxes = np.array(boxes, np.int)*2
    return boxes


def check(x_train, y_train):
    box_groups = craft_decode1(y_train)

    for i in range(len(y_train)):
        img = (x_train[i]*255).astype(np.uint8)
        img1 = img.copy()
        img2 = img.copy()
        img3 = img.copy()
        boxes1 = box_groups[i]
        
        for box in boxes1:
            p1,p2,p3,p4 = tuple(box[0].astype(int)), tuple(box[1].astype(int)), tuple(box[2].astype(int)), tuple(box[3].astype(int))  
            lines = np.array([[p1, p2], [p2, p3], [p3, p4], [p4, p1]], np.int32)
            cv2.polylines(img1, [line for line in lines], True, (0,0,255), thickness = 2)

        img = cv2.resize(img, (y_train.shape[2], y_train.shape[1]))
        boxes2 = craft_decode2(y_train[i], img)
        for box in boxes2:
            p1,p2,p3,p4 = tuple(box[0].astype(int)), tuple(box[1].astype(int)), tuple(box[2].astype(int)), tuple(box[3].astype(int))  
            lines = np.array([[p1, p2], [p2, p3], [p3, p4], [p4, p1]], np.int32)
            cv2.polylines(img2, [line for line in lines], True, (0,0,255), thickness = 2)

        boxes3 = craft_decode3(y_train[i])
        for box in boxes3:
            p1,p2,p3,p4 = tuple(box[0].astype(int)), tuple(box[1].astype(int)), tuple(box[2].astype(int)), tuple(box[3].astype(int))  
            lines = np.array([[p1, p2], [p2, p3], [p3, p4], [p4, p1]], np.int32)
            cv2.polylines(img3, [line for line in lines], True, (0,0,255), thickness = 2)
        cv2.imshow('keras-ocr', img1)
        cv2.imshow('watershed', img2)
        cv2.imshow('normal', img3)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

######################## loss plot ##########################################
csv_files = glob.glob("./losses/*.csv")


################################ load-model and test #########################
