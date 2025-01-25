import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.metrics import categorical_accuracy

def loss_ce(y_true,y_pred):
    x = -tf.math.log(y_pred + 1e-12)
    return tf.reduce_mean(y_true*x)

def loss_mse(y_true,y_pred):
    x = tf.square((y_true - y_pred))
    return tf.reduce_mean(x)

def loss_new(y_true,y_pred):
    x = tf.math.pow(tf.math.abs(y_true - y_pred),5)
    return tf.reduce_mean(x)

def accuracy(y_true,y_pred):
  #return np.mean(np.argmax(y_true,axis=-1) == np.argmax(y_pred,axis=-1))
  return tf.reduce_mean(categorical_accuracy(y_true,y_pred))