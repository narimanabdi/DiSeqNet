"""
This module generates encoders for feature extracting
"""
from tensorflow import keras
import tensorflow as tf
from tensorflow import keras
from .distances import Euclidean_Distance
from .stn import stn
from .blocks import conv_block

def create_encoder(input_shape = (32,32,3)):
    inp = keras.layers.Input(input_shape)
    x = stn(inp,filters=[200,300,200])
    x = conv_block(x,kernel_size=(7,7),n_filters=100,strides=(1,1))
    x = keras.layers.MaxPooling2D(pool_size=(3,3),strides=2)(x)

    x = conv_block(x,kernel_size=(4,4),n_filters=150,strides=(1,1))
    x = keras.layers.MaxPooling2D(pool_size=(3,3),strides=2)(x)
    x = stn(x,filters=[200,200,200])
    x = conv_block(x,kernel_size=(4,4),n_filters=250,strides=(1,1))
    x = keras.layers.MaxPooling2D(pool_size=(3,3),strides=2)(x)
    
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(300,activation='relu')(x)
    return keras.Model(inp,x)

def create_model(input_shape = (64,64,3),weights=None):
    #inp = keras.layers.Input(input_shape)
    support = keras.layers.Input(input_shape)
    query = keras.layers.Input(input_shape)
    encoder = create_encoder(input_shape=input_shape)
    if weights is not None:
        encoder.load_weights(weights)
    #x = fe(inp)
    #x = keras.layers.Dense(600)(x)
    #encoder = keras.Model(inputs=inp,outputs=x)
    support_features = encoder(support)
    query_features = encoder(query)
    dist = Euclidean_Distance()([support_features,query_features])
    out = keras.layers.Activation("softmax")(dist)
    return keras.Model(inputs = [support,query],outputs=out)
