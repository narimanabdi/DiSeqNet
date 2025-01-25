#encoder based on ResNet50
from tensorflow import keras
from tensorflow.keras.layers import Input,Flatten,Dense,MaxPooling2D,Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation
import tensorflow as tf
from .blocks import conv_block,dcp
from .distances import Weighted_Euclidean_Distance
from .stn import stn

from tensorflow.keras.applications import ResNet50


def create_resnet(input_shape = (64,64,3)):
    H,W,C = input_shape
    inp = Input([H,W,C])
    resnet = ResNet50(
        input_shape=input_shape,weights='imagenet',include_top=False)
    res_encoder = keras.Model(
        inputs=resnet.inputs,
        outputs=resnet.get_layer('conv3_block2_out').output,name='backbone')
    x = stn(inp,[64,64,100],kernel_size=(5,5),stage=1)
    x = res_encoder(x)
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
    support = Input(input_shape)
    query = Input(input_shape)
    encoder = create_resnet(input_shape=input_shape)
    support_features = encoder(support)
    query_features = encoder(query)
    dist = Weighted_Euclidean_Distance()([support_features,query_features])
    out = Activation("softmax")(dist)
    return keras.Model(inputs = [support,query],outputs=out)
