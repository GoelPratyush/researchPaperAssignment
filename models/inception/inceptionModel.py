import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import os
import sys
sys.path.append(os.path.join('..', '..'))
from global_params import *

# function for creating a projected inception module
def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
	# 1x1 conv
	conv1 = keras.layers.Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
	# 3x3 conv
	conv3 = keras.layers.Conv2D(f2_in, (1,1), padding='same', activation='relu')(layer_in)
	conv3 = keras.layers.Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)
	# 5x5 conv
	conv5 = keras.layers.Conv2D(f3_in, (1,1), padding='same', activation='relu')(layer_in)
	conv5 = keras.layers.Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)
	# 3x3 max pooling
	pool = keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
	pool = keras.layers.Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)
	# concatenate filters, assumes filters/channels last
	layer_out = keras.layers.concatenate([conv1, conv3, conv5, pool], axis=-1)
	return layer_out

def create_model():
	# define model input
	input = keras.layers.Input(shape=input_shape)

	conv1 = keras.layers.Conv2D(64, (7,7), activation='relu', strides=(2,2), padding='same', name='conv1')(input)
	maxpool1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(conv1)
	norm1 = keras.layers.BatchNormalization(name='norm1')(maxpool1)

	# add inception block 2
	conv2_1 = keras.layers.Conv2D(64, (1,1), activation='relu', strides=(1,1), padding='same', name='conv2_1')(norm1)
	conv2_3 = keras.layers.Conv2D(192, (3,3), activation='relu', strides=(1,1), padding='same', name='conv2_3')(conv2_1)
	norm2 = keras.layers.BatchNormalization(name='norm2')(conv2_3)
	maxpool2 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool2')(norm2)

	# add inception block 3a
	inception3a = inception_module(maxpool2, 64, 96, 128, 16, 32, 32)

	# add inception block 3b
	inception3b = inception_module(inception3a, 64, 96, 128, 32, 64, 64)

	# add inception block 3c
	conv3_3c = keras.layers.Conv2D(128, (1,1), padding='same', activation='relu')(inception3b)
	conv3_3c = keras.layers.Conv2D(256, (3,3), strides=(2,2), padding='same', activation='relu')(conv3_3c)
	# 5x5 conv
	conv5_3c = keras.layers.Conv2D(32, (1,1), padding='same', activation='relu')(inception3b)
	conv5_3c = keras.layers.Conv2D(64, (5,5), strides=(2,2), padding='same', activation='relu')(conv5_3c)
	# 3x3 max pooling
	pool_3c = keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(inception3b)
	# concatenate filters, assumes filters/channels last
	layer_out_3c = keras.layers.concatenate([conv3_3c, conv5_3c, pool_3c], axis=-1)

	# add inception block 4a
	inception4a = inception_module(layer_out_3c, 256, 96, 192, 32, 64, 128)

	# add inception block 4b
	inception4b = inception_module(inception4a, 224, 112, 224, 32, 64, 128)

	# add inception block 4c
	inception4c = inception_module(inception4b, 192, 128, 256, 32, 64, 128)

	# add inception block 4d
	inception4d = inception_module(inception4c, 160, 144, 288, 32, 64, 128)

	# add inception block 4e
	conv3_4e = keras.layers.Conv2D(160, (1,1), padding='same', activation='relu')(inception4d)
	conv3_4e = keras.layers.Conv2D(256, (3,3), strides=(2,2), padding='same', activation='relu')(conv3_4e)
	# 5x5 conv
	conv5_4e = keras.layers.Conv2D(64, (1,1), padding='same', activation='relu')(inception4d)
	conv5_4e = keras.layers.Conv2D(128, (5,5), strides=(2,2), padding='same', activation='relu')(conv5_4e)
	# 3x3 max pooling
	pool_4e = keras.layers.MaxPooling2D((3,3), strides=(2,2), padding='same')(inception4d)
	# concatenate filters, assumes filters/channels last
	layer_out_4e = keras.layers.concatenate([conv3_4e, conv5_4e, pool_4e], axis=-1)

	# add inception block 5a
	inception5a = inception_module(layer_out_4e, 384, 192, 384, 48, 128, 128)

	# add inception block 5b
	inception5b = inception_module(inception5a, 384, 192, 384, 48, 128, 128)

	# global average pooling layer
	avgpool = keras.layers.GlobalAveragePooling2D()(inception5b)

	# Flatten
	# flat = keras.layers.Flatten(name='flatten')(avgpool)
	drop = keras.layers.Dropout(0.2)(avgpool)
	# dense fc layer
	fc = keras.layers.Dense(128, name='fc1')(drop)
	# L2 norm
	l2norm = keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='l2')(fc)

	# create model
	return keras.Model(inputs=input, outputs=l2norm, name='NN2')