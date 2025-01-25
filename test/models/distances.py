"""
This module return distance matrix based on the diffrent distance metrics
"""
import tensorflow as tf
from tensorflow import keras

class Euclidean_Distance(keras.layers.Layer):
    def __init__(self,**kwargs):
        super(Euclidean_Distance,self).__init__(**kwargs)
        #self.constant = tf.Variable(initial_value=0.1, dtype=tf.float32, trainable=True)

    def call(self,inputs,**kwargs):
        support,query = inputs
        q2 = tf.reduce_sum(query ** 2, axis=1, keepdims=True)
        s2 = tf.reduce_sum(support ** 2, axis=1, keepdims=True)
        qdots = tf.matmul(query, tf.transpose(support))
        return  -(tf.sqrt(q2 + tf.transpose(s2) - 2 * qdots))
    
    def compute_output_shape(self,input_shape):
        return tf.TensorShape(input_shape[:-1])
    
    def get_config(self):
        config = super(Euclidean_Distance,self).get_config()
        return config
    
class Cosine_Distance(keras.layers.Layer):
    def __init__(self,**kwargs):
        super(Cosine_Distance,self).__init__(**kwargs)
        #self.constant = tf.Variable(initial_value=0.1, dtype=tf.float32, trainable=True)

    def call(self,inputs,**kwargs):
        support,query = inputs
        normalize_support = tf.nn.l2_normalize(support,1)        
        normalize_query = tf.nn.l2_normalize(query,1)
        distance = 1 - tf.matmul(normalize_query, normalize_support, transpose_b=True)
        return -distance
    
    def compute_output_shape(self,input_shape):
        return tf.TensorShape(input_shape[:-1])
    
    def get_config(self):
        config = super(Cosine_Distance,self).get_config()
        return config


class Weighted_Euclidean_Distance(keras.layers.Layer):
    def __init__(self,**kwargs):
        super(Weighted_Euclidean_Distance,self).__init__(**kwargs)
        self.alpha = tf.Variable(
            initial_value=1.0, dtype=tf.float32, trainable=True)

    def call(self,inputs,**kwargs):
        support,query = inputs
        q2 = tf.reduce_sum(query ** 2, axis=1, keepdims=True)
        s2 = tf.reduce_sum(support ** 2, axis=1, keepdims=True)
        qdots = tf.matmul(query, tf.transpose(support))
        return  -(self.alpha * (q2 + tf.transpose(s2) - 2 * qdots))
    
    def compute_output_shape(self,input_shape):
        return tf.TensorShape(input_shape[:-1])
    
    def get_config(self):
        config = super(Weighted_Euclidean_Distance,self).get_config()
        return config