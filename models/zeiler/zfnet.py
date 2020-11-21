import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import os
import sys
sys.path.append(os.path.join('..', '..'))
from global_params import *

# building zf model
def create_model():
    model = keras.Sequential([
    keras.layers.Conv2D(64, (7, 7), activation='relu', strides=(2, 2), padding='same', input_shape=input_shape, name='conv1'),
    keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1'),
    keras.layers.BatchNormalization(name='norm1'),
    
    keras.layers.Conv2D(64, (1, 1), activation='relu', strides=(1, 1), padding='same', name='conv2a'),
    keras.layers.Conv2D(192, (3, 3), activation='relu', strides=(1, 1), padding='same', name='conv2'),
    keras.layers.BatchNormalization(name='norm2'),
    keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool2'),
    
    keras.layers.Conv2D(192, (1, 1), activation='relu', strides=(1, 1), padding='same', name='conv3a'),
    keras.layers.Conv2D(384, (3, 3), activation='relu', strides=(1, 1), padding='same', name='conv3'),
    keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool3'),
    
    keras.layers.Conv2D(384, (1, 1), activation='relu', strides=(1, 1), padding='same', name='conv4a'),
    keras.layers.Conv2D(256, (3, 3), activation='relu', strides=(1, 1), padding='same', name='conv4'),
    
    keras.layers.Conv2D(256, (1, 1), activation='relu', strides=(1, 1), padding='same', name='conv5a'),
    keras.layers.Conv2D(256, (3, 3), activation='relu', strides=(1, 1), padding='same', name='conv5'),
    
    keras.layers.Conv2D(256, (1, 1), activation='relu', strides=(1, 1), padding='same', name='conv6a'),
    keras.layers.Conv2D(256, (3, 3), activation='relu', strides=(1, 1), padding='same', name='conv6'),
    keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='pool4'),

    keras.layers.Flatten(name='flatten'),
    
    keras.layers.Dense(1024, name='fc1'),
    keras.layers.Dropout(0.2, name='drop1'),
    keras.layers.Dense(1024, name='fc2'),
    keras.layers.Dropout(0.2, name='drop2'),
    keras.layers.Dense(128, name='fc3'),
    
    keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='l2')
    ], name='zeiler_fergus')
    return model