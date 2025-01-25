##################################
# Generator for GTSRB --> TT100K #
##################################
import os 

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array


from .batchgenerator import GTSRB_Generator,TT100K_Generator,standardize

def get_generator(batch=128,dim=64,test_type='all'):
    tr_gen = GTSRB_Generator(
        n_way=43,k_shot=1,batch=batch,data_type='all',
        target_size=(dim,dim))
    te_gen = TT100K_Generator(
        n_way=36,k_shot=1,batch=batch,data_type=test_type,target_size=(dim,dim),shuffle=False)
    return tr_gen,te_gen

def get_test_generator(batch=128,dim=64,type='all',shuffle = False):
    if type == 'all':
        return TT100K_Generator(
            n_way=36,k_shot=1,batch=batch,data_type='all',target_size=(dim,dim),
        shuffle=shuffle)
    else:
        return TT100K_Generator(
            n_way=32,k_shot=1,batch=batch,data_type='unseen',
            target_size=(dim,dim),shuffle=shuffle)
    
def get_test_generator_for_video():
    data_path='datasets/TT100K'
    target_size=(64,64)
    home = os.getcwd()
    original_path = os.path.join(home,data_path)
    template_path = os.path.join(original_path,"template")
    support_path = os.path.join(template_path,"all_video")
    classes = np.asarray(os.listdir(support_path))
    classes.sort()
    X = np.empty((47,*target_size,3))
    for i,C in enumerate(classes):
        c_path = os.path.join(support_path,C)
        for f in os.listdir(c_path):
            file_path = os.path.join(c_path,f)
            X[i] = standardize(img_to_array(load_img(
                file_path,target_size=target_size,
                interpolation='bilinear')))
    return tf.constant(X,dtype='float32')