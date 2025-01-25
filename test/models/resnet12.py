from tensorflow import keras
from .blocks import res_block,conv_block,dcp
from .stn import stn
from .distances import Euclidean_Distance

def create_resnet12(input_shape = (64,64,3),weights=None):
    H,W,C = input_shape
    inp = keras.layers.Input([H,W,C])
    x = inp
    x = res_block(x,kernel_size=3,n_filters=64,strides=(1,1),stage=0)
    x = res_block(x,kernel_size=3,n_filters=128,strides=(1,1),stage=1)
    x = res_block(x,kernel_size=3,n_filters=256,strides=(1,1),stage=2)
    x = res_block(x,kernel_size=3,n_filters=512,strides=(1,1),stage=3)
    #x = keras.layers.Flatten()(x)

    return keras.Model(inputs=[inp],outputs=[x])

def create_resnet(input_shape = (64,64,3)):
    H,W,C = input_shape
    inp = keras.layers.Input([H,W,C])
    resnet = create_resnet12(input_shape=input_shape)
    resnet.load_weights('model_files/pretrained_weights/res12_mini_pretrained_weights_final.h5')
    res_encoder = keras.Model(inputs=resnet.inputs,outputs=resnet.get_layer('maxpool_stage_2').output,name='backbone')
    x = res_encoder(inp)
    x = keras.layers.Conv2D(
        kernel_size=(1,1),filters=100,padding='same',kernel_initializer='he_normal')(x)
    x = stn(x,[100,100,100])
    x = conv_block(x,kernel_size=(3,3),n_filters=100,strides=(1,1))
    x = dcp(x,n_filters=100,kernel_size=(3,3))
    x = keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units = 300,kernel_initializer="he_normal")(x)
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Activation('linear')(x)
    return keras.Model(inp,x,name='encoder')

def create_model(input_shape = (64,64,3)):
    support = keras.layers.Input(input_shape)
    query = keras.layers.Input(input_shape)
    encoder = create_resnet(input_shape=input_shape)
    support_features = encoder(support)
    query_features = encoder(query)
    dist = Euclidean_Distance()([support_features,query_features])
    out = keras.layers.Activation("softmax")(dist)
    return keras.Model(inputs = [support,query],outputs=out)