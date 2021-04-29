import random
import itertools
import cv2
import numpy as np
from . import tools
from . import pad

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def get_image_data_generator(labels, width, height, augmenter=None, area_threshold=0.5, min_area=None, shuffle=True):
    """Generated augmented (image, lines) tuples from a list
    of (filepath, lines, confidence) tuples. Confidence is
    not used right now but is included for a future release
    that uses semi-supervised data.
    Args:
        labels: A list of (image, lines, confience) tuples.
        augmenter: An augmenter to apply to the images.
        area_threshold: The area threshold to use to keep characters in augmented images.
        width: The width to use for output images
        height: The height to use for output images
        min_area: The minimum area for a character to be
            included.
        focused: Whether to pre-crop images to width/height containing
            a region containing text.
        shuffle: Whether to shuffle the data on each iteration.
    """
    labels = labels.copy()
    for index in itertools.cycle(range(len(labels))):
        if index == 0 and shuffle:
            random.shuffle(labels)
        image_filepath, lines, confidence = labels[index]
        image = tools.read(image_filepath)
        if augmenter is not None:
            image, lines = tools.augment(boxes=lines, boxes_format='lines', image=image, area_threshold=area_threshold, min_area=min_area, augmenter=augmenter)

        image, scale = tools.fit(image,
                                 width=width,
                                 height=height,
                                 mode='letterbox',
                                 return_scale=True)
        lines = tools.adjust_boxes(boxes=lines, boxes_format='lines', scale=scale)
        yield image, lines, confidence


def get_gaussian_heatmap(size=512, distanceRatio=3.34):
    v = np.abs(np.linspace(-size / 2, size / 2, num=size))
    x, y = np.meshgrid(v, v)
    g = np.sqrt(x**2 + y**2)
    g *= distanceRatio / (size / 2)
    g = np.exp(-(1 / 2) * (g**2))
    g *= 255
    return g.clip(0, 255).astype('uint8')


def four_point_transform(image, pts):

	max_x, max_y = np.max(pts[:, 0]).astype(np.int32), np.max(pts[:, 1]).astype(np.int32)

	dst = np.array([
		[0, 0],
		[image.shape[1] - 1, 0],
		[image.shape[1] - 1, image.shape[0] - 1],
		[0, image.shape[0] - 1]], 
        dtype="float32"
    )

	M = cv2.getPerspectiveTransform(dst, pts)
	warped = cv2.warpPerspective(image, M, (max_x, max_y))

	return warped


class DataGenerator:
    def __init__(self, image_generator, batch_size=32):
        self.genrator = image_generator
        self.batch_size = batch_size
        self.gaussian_heatmap = get_gaussian_heatmap(512, 2.5)

    def get_item(self):
        X, y = self.__get_data()
        return X, y

    def __get_data(self):
        X = []
        Y = []       
        for i, id in enumerate(range(self.batch_size)):
            x, y, _ = next(self.genrator) 
            X.append(x)
            
            h, w, _ = x.shape
            y_ = np.zeros((h//2,w//2), np.uint8)
            for box in y:
                bbox = box[0][0]/2
                top_left = np.array([np.min(bbox[:, 0]), np.min(bbox[:, 1])]).astype(np.int32)
                bbox -= top_left[None, :]
                transformed = four_point_transform(self.gaussian_heatmap.copy(), bbox.astype(np.float32))

                start_row = max(top_left[1], 0) - top_left[1]
                start_col = max(top_left[0], 0) - top_left[0]
                end_row = min(top_left[1]+transformed.shape[0], y_.shape[0])
                end_col = min(top_left[0]+transformed.shape[1], y_.shape[1])

                y_[max(top_left[1], 0):end_row, max(top_left[0], 0):end_col] += transformed[start_row:end_row - top_left[1], start_col:end_col - top_left[0]]
            Y.append(y_)

        return np.array(X)/255.0, np.array(Y)/255.0