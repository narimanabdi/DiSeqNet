from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.metrics import Mean, CategoricalAccuracy
from models.metrics import loss_mse,accuracy,loss_new, loss_ce
import argparse
from time import time
from data_loader import get_loader
from models.distances import Weighted_Euclidean_Distance, Euclidean_Distance
from models.stn import BilinearInterpolation,Localization
from tensorflow.keras.models import load_model
from models.senet import Senet
from models.distill import Distiller
from models.student import create_model_st


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
        test_mode == 'belga2toplogo' or test_mode == 'gtsrb2tt100k':
        train_gen, val_gen,test_gen = loader.get_generator(
            batch=batch,dim=dim)
    else:
        train_gen, test_gen = loader.get_generator(
            batch=batch,dim=dim)

    return train_gen,val_gen ,test_gen
    

def meta_train(ep):
    '''
    meta-training function
    ep: the number of epochs
    '''
    student_file = 'model_files/new_2_student_' + args.test+ '_whole_s2.h5'
    student_encoder = 'model_files/new_2_student_' + args.test+ '_encoder_s2.h5'
    teacher = keras.models.load_model(
    'model_files/best_models/densenet_' + args.test+ '_whole.h5',
    custom_objects={
        'Weighted_Euclidean_Distance':Weighted_Euclidean_Distance,
        'BilinearInterpolation':BilinearInterpolation,
        'Localization':Localization,'Senet':Senet},compile=False)
    
    teacher_clone = keras.models.load_model(
    'model_files/best_models/densenet_' + args.test+ '_whole.h5',
    custom_objects={
        'Weighted_Euclidean_Distance':Weighted_Euclidean_Distance,
        'BilinearInterpolation':BilinearInterpolation,
        'Localization':Localization,'Senet':Senet},compile=False)
 
   
    optimizer_fn = keras.optimizers.Adam(learning_rate=lr,epsilon=1.0e-8)
    teacher.compile(optimizer=optimizer_fn,loss_fn=loss_mse,metrics=CategoricalAccuracy(name = 'accuracy'))
    #teacher.trainable = False
    student = create_model_st(teacher_model=teacher_clone, input_shape = (dim,dim,3))
    student.compile(optimizer=optimizer_fn,loss_fn=loss_mse,metrics=CategoricalAccuracy(name = 'accuracy'))
    #student.summary()
    best_test_acc = 0.0
    #define distiller for knowledge distillation
    #distiller = Distiller(student=student, teacher=teacher)
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=optimizer_fn,
        metrics=[keras.metrics.CategoricalAccuracy()],
        student_loss_fn=loss_mse,
        distillation_loss_fn=keras.losses.KLDivergence(),#distillation_loss_fn=keras.losses.KLDivergence()
        alpha=0.1,
        temperature=10,
    )
    train_datagen,val_datagen, test_datagen = make_data_generator(args.test)
    strat_time = time()
    for step in range(5):
        #print(f'=====step {step+1}=====')
   
        for e in range(ep):
            #print(f'=====epoch {e+1}/{ep}=====')
            distiller.fit(train_datagen,verbose=0)
            te_acc,_ = distiller.evaluate(test_datagen, verbose=0)
            if te_acc >= best_test_acc:
                best_test_acc = te_acc
                student.save(student_file)
                student.save_weights('best_weights_student.h5')
                print(f'===step {step+1} epoch {e+1}/{ep} best test accuracy: {best_test_acc:.4f}===')
            #print(f'test accuracy: {te_acc:.4f}')
            #print(f'best test accuracy: {best_test_acc:.4f}')
            
        student.load_weights('best_weights_student.h5')
        optimizer_fn.learning_rate = optimizer_fn.learning_rate * 0.5   
   
    print('Meta Training has just been ended')
    end_time = time() - strat_time
    print(f'trainig time: {end_time}')
    print(f'best test accuracy: {best_test_acc:.4f}')
    loaded_model = load_model(
        student_file,
        custom_objects={
            'Euclidean_Distance':Euclidean_Distance,
            'BilinearInterpolation':BilinearInterpolation,
            'Localization':Localization,
            'Senet':Senet},compile=False)
    enc = keras.Model(
        inputs=loaded_model.get_layer('stu_encoder').input,
        outputs=loaded_model.get_layer('stu_encoder').output)
    enc.save(student_encoder)
    print(f'cascade encoder saved at {student_encoder}')


if __name__ == "__main__":
    meta_train(args.epochs)

    