# This file contains some model architecture blocks
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation
from tensorflow.keras.layers import Concatenate,Add,MaxPooling2D

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

def conv_block_di(input_tensor,kernel_size,n_filters,dilation_rate,padding='same'):
    x = Conv2D(
        kernel_size=kernel_size,
        filters=n_filters,dilation_rate=dilation_rate,padding=padding)(input_tensor)
    x = BatchNormalization(axis=1)(x)
    return Activation('relu')(x)

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

#Block to make ResNet12
def res_block(input_tensor, kernel_size, n_filters,strides=(1,1),stage=0):
    x = Conv2D(
        n_filters, kernel_size, padding='same',kernel_initializer='he_normal',
        strides=strides)(input_tensor)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
   
    x = Conv2D(
        n_filters, kernel_size, padding='same',
        kernel_initializer='he_normal',strides=strides)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(
        n_filters, kernel_size, padding='same',
        kernel_initializer='he_normal',strides=strides)(x)
    x = BatchNormalization(axis=-1)(x)

    shortcut = Conv2D(
        n_filters, kernel_size=(1,1), padding='same',
        kernel_initializer='he_normal',strides=strides)(input_tensor)
    shortcut = BatchNormalization(axis=-1)(shortcut)


    x = Add()([x,shortcut])
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2),name='maxpool_stage_'+str(stage))(x)
    return x

#Block to make ResNet18
def res18_block(input_tensor, kernel_size, n_filters,strides=(1,1),stage=1):
    x = Conv2D(
        n_filters, kernel_size, padding='same',
        kernel_initializer='he_normal',strides=strides)(input_tensor)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('relu')(x)
    x = Conv2D(
        n_filters, kernel_size, padding='same',
        kernel_initializer='he_normal',strides=(1,1))(x)
    x = BatchNormalization(axis=-1)(x)

    shortcut = Conv2D(
        n_filters, kernel_size=(1,1), padding='same',
        kernel_initializer='he_normal',strides=strides)(input_tensor)
    shortcut = BatchNormalization(axis=-1)(shortcut)


    x = Add()([x,shortcut])
    x = Activation('relu')(name='act_'+str(stage))(x)
    return x