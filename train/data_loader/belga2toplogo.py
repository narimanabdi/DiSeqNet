####################################
# Generator for Belga --> Toplogo10#
####################################
from .batchgenerator import BELGA_Generator,FLICKR32_Generator
from .batchgenerator import TOPLOGO10_Generator

def get_generator(batch=128,dim=64):
    tr_gen = BELGA_Generator(
        n_way=37,k_shot=1,batch=batch,data_type='all',
        target_size=(dim,dim))
    va_gen = FLICKR32_Generator(
        n_way=32,k_shot=1,batch=batch,data_type='all',target_size=(dim,dim),
        shuffle=False)
    te_gen = TOPLOGO10_Generator(
        n_way=11,k_shot=1,batch=batch,data_type='all',target_size=(dim,dim),
        shuffle=False)
    return tr_gen,va_gen,te_gen

def get_test_generator(batch=128,dim=64, shuffle=False):
    return TOPLOGO10_Generator(
        n_way=11,k_shot=1,batch=batch,data_type='all',target_size=(dim,dim),
        shuffle=shuffle)