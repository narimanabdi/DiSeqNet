from tensorflow import keras
from .blocks import res18_block
from .blocks import res_block,conv_block,dcp
from .stn import stn
from .distances import Euclidean_Distance
def create_resnet18(input_shape = (64,64,3),weights=None):
    H,W,C = input_shape
    inp = keras.layers.Input([H,W,C])
    x = keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(inp)
    x = keras.layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = keras.layers.BatchNormalization(axis=1, name='bn_conv1')(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = res18_block(x,kernel_size=3,n_filters=64,strides=(1,1),stage=1)
    x = res18_block(x,kernel_size=3,n_filters=64,strides=(1,1),stage=2)


    x = res18_block(x,kernel_size=3,n_filters=128,strides=(2,2),stage=3)
    x = res18_block(x,kernel_size=3,n_filters=128,strides=(1,1),stage=4)


    x = res18_block(x,kernel_size=3,n_filters=256,strides=(2,2),stage=5)
    x = res18_block(x,kernel_size=3,n_filters=256,strides=(1,1),stage=6)


    x = res18_block(x,kernel_size=3,n_filters=512,strides=(2,2),stage=7)
    x = res18_block(x,kernel_size=3,n_filters=512,strides=(1,1),stage=8)

    #x = keras.layers.Flatten()(x)

    return keras.Model(inputs=[inp],outputs=[x])

def create_resnet(input_shape = (64,64,3)):
    H,W,C = input_shape
    inp = keras.layers.Input([H,W,C])
    resnet = create_resnet18(input_shape=input_shape)
    resnet.summary()
    resnet.load_weights('model_files/pretrained_weights/res18_mini_last.h5')
    res_encoder = keras.Model(inputs=resnet.inputs,outputs=resnet.get_layer('act_4').output,name='backbone')
    x = res_encoder(inp)
    x = keras.layers.Conv2D(
        kernel_size=(1,1),filters=100,padding='same',kernel_initializer='he_normal')(x)
    x = stn(x,[128,128,100])
    x = conv_block(x,kernel_size=(3,3),n_filters=128,strides=(1,1))
    x = dcp(x,n_filters=128,kernel_size=(3,3))
    x = keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=300,activation='linear',kernel_initializer="he_normal")(x)
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