# This module generates encoders for feature extracting
from tensorflow import keras
from tensorflow.keras.layers import Input,Flatten,Dense,MaxPooling2D,Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout
from .blocks import conv_block
from .distances import Weighted_Euclidean_Distance
from .stn import stn
from tensorflow.keras.applications import DenseNet121
from models.senet import Senet

def create_densenet(input_shape = (64,64,3)):
    inp = Input(input_shape)
    
    densnet = DenseNet121(
        input_shape=input_shape,weights='imagenet',include_top=False)
    dens_encoder = keras.Model(
        inputs=densnet.inputs,
        outputs=densnet.get_layer('conv3_block2_concat').output)
    x = stn(inp,[64,64,100],kernel_size=(5,5),stage=1)
    x = dens_encoder(x)
    x = Conv2D(
        kernel_size=(1,1),filters=100,padding='same',
        kernel_initializer='he_normal')(x)
    x = stn(x,[100,100,100], stage=2)
    x = conv_block(x,kernel_size=(3,3),n_filters=100,strides=(1,1))
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    x = Flatten()(x)
    x = Dense(units = 300,kernel_initializer="he_normal")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('linear')(x)
    return keras.Model(inp,x,name='encoder')

def create_model(input_shape = (64,64,3)):
    support = Input(input_shape)
    query = Input(input_shape)
    encoder = create_densenet(input_shape=input_shape)
    support_features = encoder(support)
    query_features = encoder(query)
    dist = Weighted_Euclidean_Distance()([support_features,query_features])
    out = Activation("softmax")(dist)
    return Senet(inputs = [support,query],outputs=out)
