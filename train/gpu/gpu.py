import tensorflow as tf
def GPU_info():
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
                print("Device Name: ", tf.config.experimental.get_device_details(gpu)['device_name'])
                print("Compute Capability: ", tf.config.experimental.get_device_details(gpu)['compute_capability'])
                print("Device Policy: ",tf.config.experimental.get_device_policy())
                print("Memory Growth: ",tf.config.experimental.get_memory_growth(gpu))
        except RuntimeError as e:
            print(e)
    else:
        print('GPU not found')

def set_gpu_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    print('GPU memory set growth')