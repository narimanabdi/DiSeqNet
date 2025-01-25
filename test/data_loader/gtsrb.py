##################################
# Generator for GTSRB --> GTSRB  #
##################################
from .batchgenerator import GTSRB_Generator

def get_generator(batch=128,dim=64):
    tr_gen = GTSRB_Generator(
        n_way=22,k_shot=1,batch=batch,data_type='seen',
        target_size=(dim,dim))
    va_gen = GTSRB_Generator(
        n_way=21,k_shot=1,batch=batch,data_type='unseen',target_size=(dim,dim),
        shuffle=False)
    te_gen = GTSRB_Generator(
        n_way=43,k_shot=1,batch=batch,data_type='all',target_size=(dim,dim),
        shuffle=False)
    return tr_gen,va_gen,te_gen

def get_test_generator(batch=128,dim=64,shuffle=False):
    return GTSRB_Generator(
        n_way=43,k_shot=1,batch=batch,data_type='all',target_size=(dim,dim),
        shuffle=shuffle)