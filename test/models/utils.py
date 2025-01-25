from models import densenet,resnet,mobilenet,densenetmini
from tensorflow.keras.applications import DenseNet121
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.metrics import categorical_accuracy
import numpy as np
from keras import backend as K

def make_senet_model(backbone,input_shape):
    """
    This function make the senet model based on different backbones
    ---------
    Args:
    resent: ResNet50
    densenet: DenseNet121\n
    mobilenet: MobileNet V2
    densenetmini: DenseNet121 pre-trained on miniImageNet
    """

    if backbone == 'resnet':
        return resnet.create_model(input_shape = input_shape)
    if backbone == 'densenet':
        return densenet.create_model(input_shape = input_shape)
    if backbone == 'mobilenet':
        return mobilenet.create_model(input_shape = input_shape)
    if backbone == 'densenetmini':
        return densenetmini.create_model(input_shape = input_shape)


def make_base_model(backbone,n_class,dim):
    """
    make base model for base traning phase
    """
    inp = keras.layers.Input((dim,dim,3))
    if backbone == 'densenet':
        encoder = DenseNet121(
            input_shape=(dim,dim,3),weights=None,include_top=False)
 
    x = encoder(inp)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(2048,activation='relu')(x)
    x = keras.layers.Dense(1024,activation='relu')(x)
    x = keras.layers.Dense(n_class,activation='softmax')(x)
    
    return keras.Model(inp,x),encoder

def loss_ce(y_true,y_pred):
    """The Cross-Entropy loss function"""
    x = -tf.math.log(y_pred + 1e-12)
    return tf.reduce_mean(y_true*x)

def loss_mse(y_true,y_pred):
    """ The MSE loss function"""
    x = tf.square((y_true - y_pred))
    return tf.reduce_mean(x)

def accuracy(y_true,y_pred):
  #return np.mean(np.argmax(y_true,axis=-1) == np.argmax(y_pred,axis=-1))
  return tf.reduce_mean(categorical_accuracy(y_true,y_pred))

def count_params(model):
    '''
    Return the number of model parameters
    '''
    params = (
        np.sum([K.count_params(p) for p in model.trainable_weights]) + 
        np.sum([K.count_params(p) for p in model.non_trainable_weights])) / 1e6
    return params