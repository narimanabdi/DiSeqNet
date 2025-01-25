from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.metrics import Mean, CategoricalAccuracy
from models.metrics import loss_mse,accuracy,loss_new, loss_ce
from models.makemodels import make_proto_model
import argparse
from time import time
from data_loader import get_loader
from models.distances import Weighted_Euclidean_Distance, Euclidean_Distance
from models.stn import BilinearInterpolation,Localization
from tensorflow.keras.models import load_model
from models.senet import Senet
from models._distill import Distiller
from models.student3 import create_model_st


parser = argparse.ArgumentParser('SENet')
parser.add_argument('--test',type=str,default='gtsrb2tt100k')
parser.add_argument('--epochs',type=int,default=50)
parser.add_argument('--batch',type=int,default=128)
parser.add_argument('--dim',type=int,default=64)
parser.add_argument('--lr',type=float,default=1e-3)
args = parser.parse_args()
#trainng parameters
dim = args.dim
batch = args.batch
epochs = args.epochs
lr = args.lr
#metric trackers
train_acc_tracker = Mean('train_accuracy')
train_loss_tracker = Mean('train_loss')
test_acc_tracker = Mean('train_accuracy')

def make_data_generator(test_mode):
    '''
    return train & test data loaders
    '''
    loader = get_loader(test_mode) 
    if test_mode== 'belga2flick' or \
        test_mode == 'belga2toplogo' or test_mode == 'gtsrb':
        train_datagen, val_datagen,test_datagen = loader.get_generator(
            batch=batch,dim=dim)
    else:
        train_datagen, test_datagen = loader.get_generator(
            batch=batch,dim=dim)

    return train_datagen ,test_datagen

def make_teacher_encoder():
    encoder = keras.models.load_model(
    'model_files/best_encoders/densenet_' + args.test+ '_encoder.h5',
    custom_objects={
        'BilinearInterpolation':BilinearInterpolation,
        'Localization':Localization},compile=False)
    support = keras.layers.Input((64,64,3))
    query = keras.layers.Input((64,64,3))
    support_features = encoder(support)
    query_features = encoder(query)
    dist = Euclidean_Distance()([support_features,query_features])
    return Senet(inputs = [support,query],outputs=dist)
    

def meta_train(ep,alpha,temp):
    '''
    meta-training function
    ep: the number of epochs
    '''
  
    te = make_teacher_encoder()
   
    optimizer_fn = keras.optimizers.Adam(learning_rate=lr,epsilon=1.0e-8)

    te.compile(optimizer=optimizer_fn,loss_fn=loss_mse,metrics=CategoricalAccuracy(name = 'accuracy'))
    student = create_model_st(input_shape = (dim,dim,3))
    student.compile(optimizer=optimizer_fn,loss_fn=loss_mse,metrics=CategoricalAccuracy(name = 'accuracy'))
    best_test_acc = 0.0
    #define distiller for knowledge distillation

    distiller = Distiller(student=student, teacher=te)
    distiller.compile(
        optimizer=optimizer_fn,
        metrics=[keras.metrics.CategoricalAccuracy()],
        student_loss_fn=loss_mse,
        distillation_loss_fn=keras.losses.KLDivergence(),#distillation_loss_fn=keras.losses.KLDivergence()
        alpha=alpha,
        temperature=temp,
    )
    train_datagen,test_datagen = make_data_generator(args.test)

    for e in range(ep):
        print(f'=====epoch {e+1}/{ep}=====')
        distiller.fit(train_datagen)
        te_acc,_ = distiller.evaluate(test_datagen, verbose=0)
        test_acc_tracker.update_state(te_acc)
    
    print(f'test accuracy: {test_acc_tracker.result():.4f} alpha: {alpha} Temperature: {temp}')


if __name__ == "__main__":
    alphas = [0.1, .2, .5]
    temperatures = [10., 3.]
    for alpha in alphas:
        for temp in temperatures:
            meta_train(1,alpha=alpha,temp=temp)

    