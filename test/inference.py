##############################
#                            #
# Nearest Neighbor Inference #
#                            #
##############################

from time import time
from data_loader import get_loader
from tensorflow.keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Mean
from prettytable import PrettyTable
from utils import count_params
import argparse
from models.distances import Cosine_Distance, Euclidean_Distance

parser = argparse.ArgumentParser('Nearest Neighbor Test')
parser.add_argument('--data',type = str,default='gtsrb2tt100k',help = 'Test type')
parser.add_argument('--mode',type = str,default='normal')
parser.add_argument('--batch',type = int,default=128)
parser.add_argument('--metric',default='l2')
parser.add_argument('--lite',help='TFLite Encoder File name')
args = parser.parse_args()

#get data loader
loader = get_loader(args.data) 
batch = args.batch
#test_generator = loader.get_test_generator(batch=batch,dim=64,shuffle=False)
test_generator = loader.get_test_generator(batch=batch,dim=64,shuffle=False)
#tracker for benckmarking
acc_tracker = Mean(name='Nearest_Neighbor_Accuracy')
time_tracker = Mean(name='Time')
@tf.function
def nn(model,inp,ztemplates):
    z = model(tf.expand_dims(inp,axis=0))
    return Euclidean_Distance()([ztemplates,z])

@tf.function
def nn_lite(inp,ztemplates):
    z = tf.expand_dims(inp,axis=0)
    if args.metric == 'l2':
        return Euclidean_Distance()([ztemplates,z])
    elif args.metric == 'cosine':
        return Cosine_Distance()([ztemplates,z])

def inference_lite(interpreter,img,input_details,output_details):
    '''
    This function genererate output for a single input image 
    '''
    img = tf.expand_dims(img,axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])


def run_model(original_encoder_file,test='gtsrb2tt100k'):
    '''
    Run a standard test
    original_encoder_file: Encoder h5 file
    '''
    #generate data loader
    original_encoder = load_model(original_encoder_file)
    original_encoder.summary()
    loader = get_loader(test) 
    #test_generator = loader.get_test_generator(batch=batch,dim=32,shuffle=False)
    test_generator = loader.get_test_generator(batch=batch,dim=64,shuffle=False)
    #test_generator = tt100k_generator
    #extract template image to fit nearest neighbor
    t = iter(test_generator)
    [Xs,_Xq],_y = next(t)
    del _Xq,_y,t
    #number of classes
    n_cls = len(Xs)
    #embed Xs to latent space
    Zs = original_encoder(Xs)
    #fit nearest neighbor to latent space
    #nn = KNeighborsClassifier(n_neighbors=1,n_jobs=-1,metric = args.metric)
    y_train = to_categorical(np.arange(n_cls),num_classes=n_cls)
    #nn.fit(Zs,y_train);
    #start inference and generate report
    print('\033[0;32mStart Nearest Neighbor Test\033[0m')
    start_time = time()
    batches = 0
    Zq = np.empty((args.batch,100))
    tval = []
    pb = tf.keras.utils.Progbar(len(test_generator),verbose=1)
    p = 0
    for data,y_test in test_generator:
        for i,x in enumerate(data[1]):
            s = time()
            #Zq[i] = original_encoder(tf.expand_dims(x,axis=0))
            #acc_tracker.update_state(nn.score(tf.expand_dims(Zq[i],axis=0),tf.expand_dims(y_test[i],axis=0)))
            #p = nn.predict(tf.expand_dims(Zq[i],axis=0))
            p = nn(original_encoder,x,Zs)
            tval.append(time() - s)
            if np.argmax(p) == np.argmax(y_test[i]):
                acc_tracker.update_state(1.0)
            else:
                acc_tracker.update_state(0.0)
        #acc_tracker.update_state(nn.score(Zq,y_test))
        batches = batches + 1
        #break
        pb.add(1)
    end_time = time()
    #fps = 1.0 / ((end_time - start_time) / (batches * args.batch))
    tval = np.asarray(tval)
    tmean = tf.math.reduce_mean(tval)
    tstd = tf.math.reduce_std(tval)
    fps = 1.0 / tmean
    myTable = PrettyTable([" 1-NN Testing Report", ""])
    myTable.add_row(["Evaluation", test])
    myTable.add_row(["Mean Accuracy", f'{acc_tracker.result()*100.0:.2f}'])
    myTable.add_row(["Model Parameters", f'{count_params(original_encoder):.2f}M'])
    myTable.add_row(["FPS", f'{fps:.1f}'])
    myTable.add_row(["Average Inference Time", f'{tmean*1000:.1f}ms'])
    myTable.add_row(["Inference Time Std", f'{tstd*1000:.1f}ms'])
    print('\033[0;31m')
    print(myTable)
    print('\033[0m')

def run_model_lite(original_encoder_file,lite_encoder_file,test='gtsrb2tt100k'):
    '''
    Run a standard test
    original_encoder_file: Encoder h5 file
    lite_encoder_file: tflite encoder file
    '''
    #generate data loader
    original_encoder = load_model(original_encoder_file)
    loader = get_loader(test) 
    test_generator = loader.get_test_generator(batch=batch,dim=64,shuffle=False)
    #extract template image to fit nearest neighbor
    t = iter(test_generator)
    [Xs,_Xq],_y = next(t)
    del _Xq,_y,t
    #number of classes
    n_cls = len(Xs)
    #embed Xs to latent space
    Zs = original_encoder(Xs)
    #fit nearest neighbor to latent space
    #nn = KNeighborsClassifier(n_neighbors=1,n_jobs=-1,metric = args.metric)
    y_train = to_categorical(np.arange(n_cls),num_classes=n_cls)
    #nn.fit(Zs,y_train);
    #start inference and generate report
    interpreter = tf.lite.Interpreter(lite_encoder_file)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print('\033[0;32mStart Nearest Neighbor Test\033[0m')
    start_time = time()
    batches = 0
    Zq = np.empty((args.batch,100))
    pb = tf.keras.utils.Progbar(len(test_generator),verbose=1,stateful_metrics=['train loss','train acc'])
    tval = []
    p = []
    for data,y_test in test_generator:
        
        for i,x in enumerate(data[1]):
            s = time()
            Zq[i] = inference_lite(
                interpreter = interpreter,
                img= x,
                input_details=input_details,
                output_details=output_details)
            #acc_tracker.update_state(nn.score(tf.expand_dims(Zq[i],axis=0),tf.expand_dims(y_test[i],axis=0)))
            #p = nn.predict(tf.expand_dims(Zq[i],axis=0))
            p = nn_lite(inp=Zq[i],ztemplates=Zs)
            tval.append(time() - s)
            #acc_tracker.update_state(nn.score(tf.expand_dims(p,axis=0),tf.expand_dims(y_test[i],axis=0)))
            if np.argmax(p) == np.argmax(y_test[i]):
                acc_tracker.update_state(1.0)
            else:
                acc_tracker.update_state(0.0)
            #Zq[i] = stu(tf.expand_dims(x,axis=0))
        #acc_tracker.update_state(nn.score(Zq,y_test))
        #break
        batches = batches + 1
        pb.add(1)
    end_time = time()
    #fps = 1.0 / ((end_time - start_time) / (batches * args.batch))
    tval = np.asarray(tval)
    tmean = tf.math.reduce_mean(tval)
    tstd = tf.math.reduce_std(tval)
    fps = 1.0 / tmean
    myTable = PrettyTable([" 1-NN Testing Report", ""])
    myTable.add_row(["Evaluation", test])
    myTable.add_row(["Mean Accuracy", f'{acc_tracker.result()*100.0:.2f}'])
    myTable.add_row(["Model Parameters", f'{count_params(original_encoder):.2f}M'])
    myTable.add_row(["FPS", f'{fps:.1f}'])
    myTable.add_row(["Average Inference Time", f'{tmean*1000:.1f}ms'])
    myTable.add_row(["Inference Time Std", f'{tstd*1000:.1f}ms'])
    print('\033[0;31m')
    print(myTable)
    print('\033[0m')



if __name__ == '__main__':
    if args.mode == 'lite':
        #run_model_lite(
                #original_encoder_file='model_files/best_encoders/student_' + args.data + '_encoder.h5',
                #lite_encoder_file='model_files/best_encoders/student_' + args.data + '_encoder.tflite',
                #test=args.data)
        run_model_lite(
                original_encoder_file='model_files/best_encoders/new_2_student_gtsrb2tt100k_encoder_s2.h5',
                lite_encoder_file='model_files/best_encoders/new_2_student_gtsrb2tt100k_encoder_s2.tflite',
                test=args.data)
    else:
        run_model(
            #original_encoder_file='model_files/best_encoders/student_' + args.data + '_encoder.h5',
            original_encoder_file= 'model_files/best_encoders/new_2_student_gtsrb2tt100k_encoder_s2.h5',
            test=args.data)
