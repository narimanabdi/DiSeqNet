####################################
# Generator for GTSRB --> FLICKR32 #
####################################
from .batchgenerator import GTSRB_Generator,FLICKR32_Generator

def get_generator(batch=128,dim=64):
    tr_gen = GTSRB_Generator(
        n_way=43,k_shot=1,batch=batch,data_type='all',
        target_size=(dim,dim))
    te_gen = FLICKR32_Generator(
        n_way=32,k_shot=1,batch=batch,data_type='all',target_size=(dim,dim),
        shuffle=False)
    return tr_gen,te_gen

def get_test_generator(batch=128,dim=64, shuffle=False):
    return FLICKR32_Generator(
        n_way=32,k_shot=1,batch=batch,data_type='all',target_size=(dim,dim),
        shuffle=shuffle)
