'''
Convert H5 encoder to quantized TFlite
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import argparse
parser = argparse.ArgumentParser('TFLite Converter')
parser.add_argument('--source',type=str)
parser.add_argument('--target',type=str)
args = parser.parse_args()

def convert(file,qfile):
    '''
    file: original H5 model filename
    qfile: quantized TFlite model filename
    '''
    dir_path = 'model_files/'
    file_path = os.path.join(dir_path,file)
    qfile_path = os.path.join(dir_path,qfile)
    model = keras.models.load_model(file_path,compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    with open(qfile_path, 'wb') as f:
        f.write(tflite_quant_model)
    f.close()
    print('ًQuantization is done')
    

if __name__ == "__main__":
    convert(args.source,args.target)
