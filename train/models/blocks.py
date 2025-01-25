# This file contains some model architecture blocks
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation
from tensorflow.keras.layers import Concatenate

#Convolutional block with BachNormalization and ReLU activation function
def conv_block(input_tensor,kernel_size,n_filters,strides,padding='valid'):
    x = Conv2D(
        kernel_size=kernel_size,
        filters=n_filters,
        padding=padding,strides=strides,
        kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    return x
def conv_block1x1(input_tensor,kernel_size,n_filters,strides,padding='valid'):
    x = Conv2D(
        kernel_size=(1,1),
        filters=n_filters,
        kernel_initializer='he_normal')(input_tensor)
    x = Conv2D(
        kernel_size=kernel_size,
        filters=n_filters,
        padding=padding,strides=strides,
        kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    return x

# Dilated Convolution Pyramid (DCP) block
def dcp(input_tensor,kernel_size,n_filters):
    x1 = Conv2D(
        kernel_size=kernel_size,filters=n_filters,padding='same',
        dilation_rate=(1,1),kernel_initializer='he_normal')(input_tensor)
    
    
    x2 = Conv2D(
        kernel_size=kernel_size,filters=n_filters,padding='same',
        dilation_rate=(2,2),kernel_initializer='he_normal')(input_tensor)

    
    x = Concatenate()([x1,x2])
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    return x