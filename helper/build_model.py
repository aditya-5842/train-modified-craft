import numpy as np 
from imgaug.imgaug import pool
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.layers.convolutional import UpSampling1D
from tensorflow.python.training.tracking import base
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow import keras



def build_effiecient_basenetmodel(basenet_trainable=True):
    # backbone (reducing size)
    basenet = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(None, None, 3))
    if not basenet_trainable:
        basenet.trainable = False
    basenet = tf.keras.Model(basenet.input, basenet.layers[-4].output)
    layers = ["block1a_project_conv", "block2b_add", "block3b_add", "block4c_add", "block5c_add", "block6d_add", "block7a_project_bn"]
    s0, s1, s2, s3, s4, s5, s6 = [basenet.get_layer(layer).output for layer in layers]
    s3 = tf.keras.layers.Concatenate()([s3, s4])
    s4 = tf.keras.layers.Concatenate()([s5, s6])
    x = upconv(s4, 1, 512)
    x = UpsampleLike()([x, s3])

    x = tf.keras.layers.Concatenate()([x, s3])

    x = upconv(x, 2, 256)
    x = UpsampleLike()([x, s2])

    x = tf.keras.layers.Concatenate()([x, s2])

    x = upconv(x, 3, 128)
    x = UpsampleLike()([x, s1])

    x = tf.keras.layers.Concatenate()([x, s1])

    x = upconv(x, 4, 64)
    x = UpsampleLike()([x, s0])

    y = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', name='conv_cls.0')(x)
    y = keras.layers.Activation('relu', name='conv_cls.1')(y)
    y = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', name='conv_cls.2')(y)
    y = keras.layers.Activation('relu', name='conv_cls.3')(y)
    y = keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', name='conv_cls.4')(y)
    y = keras.layers.Activation('relu', name='conv_cls.5')(y)
    y = keras.layers.Conv2D(filters=16, kernel_size=1, strides=1, padding='same', name='conv_cls.6')(y)
    y = keras.layers.Activation('relu', name='conv_cls.7')(y)
    y = keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='same', name='conv_cls.8')(y)
    y = keras.layers.Activation('sigmoid')(y)
    model = keras.models.Model(inputs=basenet.input, outputs=y, name='craft_with_efficientnet_as_basenet')

    return model

def upconv(x, n, filters):
    x = keras.layers.Conv2D(filters=filters, kernel_size=1, strides=1, name=f'upconv{n}.conv.0')(x)
    x = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=f'upconv{n}.conv.1')(x)
    x = keras.layers.Activation('relu', name=f'upconv{n}.conv.2')(x)
    x = keras.layers.Conv2D(filters=filters // 2, kernel_size=3, strides=1, padding='same', name=f'upconv{n}.conv.3')(x)
    x = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name=f'upconv{n}.conv.4')(x)
    x = keras.layers.Activation('relu', name=f'upconv{n}.conv.5')(x)
    return x


class UpsampleLike(keras.layers.Layer):
    """ Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    """
    # pylint:disable=unused-argument
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        if keras.backend.image_data_format() == 'channels_first':
            raise NotImplementedError
        else:
            # pylint: disable=no-member
            return tf.compat.v1.image.resize_bilinear(source, size=(target_shape[1], target_shape[2]), half_pixel_centers=True)
    def compute_output_shape(self, input_shape):
        if keras.backend.image_data_format() == 'channels_first':
            raise NotImplementedError
        else:
            return (input_shape[0][0], ) + input_shape[1][1:3] + (input_shape[0][-1], )


def make_vgg_block(x, filters, n, prefix, pooling=True):
    x = keras.layers.Conv2D(filters=filters, strides=(1, 1), kernel_size=(3, 3), padding='same', name=f'{prefix}.{n}')(x)
    x = keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, axis=-1, name=f'{prefix}.{n+1}')(x)
    x = keras.layers.Activation('relu', name=f'{prefix}.{n+2}')(x)
    if pooling:
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='valid', strides=(2, 2), name=f'{prefix}.{n+3}')(x)
    return x

def build_vgg_backbone(inputs):
    x = make_vgg_block(inputs, filters=64, n=0, pooling=False, prefix='basenet.slice1')
    x = make_vgg_block(x, filters=64, n=3, pooling=True, prefix='basenet.slice1')
    x = make_vgg_block(x, filters=128, n=7, pooling=False, prefix='basenet.slice1')
    x = make_vgg_block(x, filters=128, n=10, pooling=True, prefix='basenet.slice1')
    x = make_vgg_block(x, filters=256, n=14, pooling=False, prefix='basenet.slice2')
    x = make_vgg_block(x, filters=256, n=17, pooling=False, prefix='basenet.slice2')
    x = make_vgg_block(x, filters=256, n=20, pooling=True, prefix='basenet.slice3')
    x = make_vgg_block(x, filters=512, n=24, pooling=False, prefix='basenet.slice3')
    x = make_vgg_block(x, filters=512, n=27, pooling=False, prefix='basenet.slice3')
    x = make_vgg_block(x, filters=512, n=30, pooling=True, prefix='basenet.slice4')
    x = make_vgg_block(x, filters=512, n=34, pooling=False, prefix='basenet.slice4')
    x = make_vgg_block(x, filters=512, n=37, pooling=False, prefix='basenet.slice4')
    x = make_vgg_block(x, filters=512, n=40, pooling=True, prefix='basenet.slice4')
    vgg = keras.models.Model(inputs=inputs, outputs=x)
    return [
        vgg.get_layer(slice_name).output for slice_name in [
            'basenet.slice1.12',
            'basenet.slice2.19',
            'basenet.slice3.29',
            'basenet.slice4.38',
        ]
    ]

def build_keras_model():
    inputs = keras.layers.Input((None, None, 3))

    s1, s2, s3, s4 = build_vgg_backbone(inputs)

    s5 = keras.layers.MaxPooling2D(pool_size=3, strides=1, padding='same', name='basenet.slice5.0')(s4)
    s5 = keras.layers.Conv2D(1024, kernel_size=(3, 3), padding='same', strides=1, dilation_rate=6, name='basenet.slice5.1')(s5)
    s5 = keras.layers.Conv2D(1024, kernel_size=1, strides=1, padding='same', name='basenet.slice5.2')(s5)

    y = keras.layers.Concatenate()([s5, s4])
    y = upconv(y, n=1, filters=512)
    y = UpsampleLike()([y, s3])
    y = keras.layers.Concatenate()([y, s3])
    y = upconv(y, n=2, filters=256)
    y = UpsampleLike()([y, s2])
    y = keras.layers.Concatenate()([y, s2])
    y = upconv(y, n=3, filters=128)
    y = UpsampleLike()([y, s1])
    y = keras.layers.Concatenate()([y, s1])
    features = upconv(y, n=4, filters=64)

    y = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', name='conv_cls.0')(features)
    y = keras.layers.Activation('relu', name='conv_cls.1')(y)
    y = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', name='conv_cls.2')(y)
    y = keras.layers.Activation('relu', name='conv_cls.3')(y)
    y = keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', name='conv_cls.4')(y)
    y = keras.layers.Activation('relu', name='conv_cls.5')(y)
    y = keras.layers.Conv2D(filters=16, kernel_size=1, strides=1, padding='same', name='conv_cls.6')(y)
    y = keras.layers.Activation('relu', name='conv_cls.7')(y)
    y = keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='same', name='conv_cls.8')(y)
    y = keras.layers.Activation('sigmoid')(y)

    model = keras.models.Model(inputs=inputs, outputs=y)
    return model

def load_keras_model_vgg16(basenet_trainable=True):
    model = build_keras_model()

    #load vgg-16 model
    vgg16 = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(None, None, 3))

    
    model_layers_name = [
        "{prefix}.{n}".format(filters=64, n=0, prefix='basenet.slice1'),
        "{prefix}.{n}".format(filters=64, n=3, prefix='basenet.slice1'),
        "{prefix}.{n}".format(filters=128, n=7, prefix='basenet.slice1'),
        "{prefix}.{n}".format(filters=128, n=10, prefix='basenet.slice1'),
        "{prefix}.{n}".format(filters=256, n=14, prefix='basenet.slice2'),
        "{prefix}.{n}".format(filters=256, n=17, prefix='basenet.slice2'),
        "{prefix}.{n}".format(filters=256, n=20, prefix='basenet.slice3'),
        "{prefix}.{n}".format(filters=512, n=24, prefix='basenet.slice3'),
        "{prefix}.{n}".format(filters=512, n=27, prefix='basenet.slice3'),
        "{prefix}.{n}".format(filters=512, n=30, prefix='basenet.slice4'),
        "{prefix}.{n}".format(filters=512, n=34, prefix='basenet.slice4'),
        "{prefix}.{n}".format(filters=512, n=37, prefix='basenet.slice4'),
        "{prefix}.{n}".format(filters=512, n=40, prefix='basenet.slice4')
    ]

    vgg_layers_name = [
        "block1_conv1",      
        "block1_conv2",     
        "block2_conv1",     
        "block2_conv2",    
        "block3_conv1",    
        "block3_conv2",    
        "block3_conv3",    
        "block4_conv1",   
        "block4_conv2",   
        "block4_conv3",   
        "block5_conv1",   
        "block5_conv2",   
        "block5_conv3"
    ]

    model_layers = [layer for layer in model.layers if layer.name in model_layers_name]
    vgg16_layers = [layer for layer in vgg16.layers if layer.name in vgg_layers_name]
    
    for model_layer, vgg_layer in zip(model_layers, vgg16_layers):
        model_layer.weights[0].assign(vgg_layer.weights[0])
        model_layer.weights[1].assign(vgg_layer.weights[1])

    if not basenet_trainable:
        # all basenet convnet layers trainable paraemter are going to be set to non tainable
        for layer in model.layers:
            if layer.name in model_layers_name:
                layer.trainable = False
    return model

