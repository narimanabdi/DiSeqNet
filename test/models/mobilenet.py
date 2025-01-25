"""
This module generates encoders for feature extracting
"""
from tensorflow import keras
from tensorflow.keras.layers import Input,Flatten,Dense,MaxPooling2D,Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation
import tensorflow as tf
from .blocks import conv_block,dcp
from .distances import Weighted_Euclidean_Distance
from .stn import stn

from tensorflow.keras.applications import  MobileNetV2

def create_mobilenet(input_shape = (64,64,3)):
    inp = keras.layers.Input(input_shape)
    mobilenet = MobileNetV2(input_shape=input_shape,weights='imagenet',include_top=False)
    mob_encoder = keras.Model(inputs=mobilenet.inputs,outputs=mobilenet.get_layer('block_5_add').output)
    mob_encoder.summary()
    x = stn(inp,[64,64,100],kernel_size=(5,5),stage=1)
    x = mob_encoder(x)
    x = Conv2D(
        kernel_size=(1,1),filters=100,padding='same',
        kernel_initializer='he_normal')(x)
    x = stn(x,[100,100,100],stage=2)
    x = conv_block(x,kernel_size=(3,3),n_filters=100,strides=(1,1))
    x = dcp(x,n_filters=100,kernel_size=(3,3))
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    x = Flatten()(x)
    x = Dense(units = 300,kernel_initializer="he_normal")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('linear')(x)
    return keras.Model(inp,x,name='encoder')

def create_model(input_shape = (64,64,3)):
    support = keras.layers.Input(input_shape)
    query = keras.layers.Input(input_shape)
    encoder = create_mobilenet(input_shape=input_shape)
    encoder.summary()
    support_features = encoder(support)
    query_features = encoder(query)
    dist = Weighted_Euclidean_Distance()([support_features,query_features])
    out = keras.layers.Activation("softmax")(dist)
    return keras.Model(inputs = [support,query],outputs=out)
