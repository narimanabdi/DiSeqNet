import pandas as pd
import os
from PIL import Image, ImageDraw, ImageFont, ImageFile
import numpy as np
import matplotlib.pyplot as plt
from time import time

import cv2
from tqdm import tqdm
import tensorflow as tf
from data_loader import get_loader
from tensorflow import keras
from models.stn import BilinearInterpolation,Localization
from models.distances import Euclidean_Distance
import json

json_file = open('cfgs/test_video.json')
config = json.load(json_file)



def normalize(img):
    return (img - np.min(img))/(np.max(img) - np.min(img))

def read_input_frame(frame_idx,data,folder_name):
    data_path = "datasets/CCTSDB"
    image_folder = os.path.join(data_path,folder_name)
    image_path = os.path.join(image_folder,data['file_name'][frame_idx])
    image = Image.open(image_path).convert("RGB")
    bounding_box = (data['x_min'][frame_idx], data['y_min'][frame_idx], data['x_max'][frame_idx], data['y_max'][frame_idx])
    return image, bounding_box

def extract_sign(image,bbox,target_size):
    cropped_image = image.crop(bbox)
    resized_image = np.array(cropped_image.resize(target_size))
    return resized_image



class Inference:
    def __init__(self, encoder_h5,lite_encoder_file, template_set):
        self.loaded_encoder = keras.models.load_model(
            encoder_h5,
            custom_objects={
                'BilinearInterpolation': BilinearInterpolation,
                'Localization': Localization}, compile=False)
        self.Xs = template_set
        self.Zs = self.loaded_encoder(self.standardize(self.Xs))
        self.dist_fn = Euclidean_Distance()
        self.interpreter = tf.lite.Interpreter(lite_encoder_file)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    @tf.function
    def standardize(self,img):
        img = tf.cast(img, dtype=tf.float32)
        mean = tf.math.reduce_mean(img)
        std = tf.math.reduce_std(img)
        return (img - mean) / std

    @tf.function
    def nn(self,inp,ztemplates):
        return self.dist_fn([ztemplates, inp])

    def inference_lite(self,img):
        '''
        This function genererate output for a single input image 
        '''
        img = tf.expand_dims(img,axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])

    def __call__(self, input_image):
        x = self.standardize(input_image)
        out = self.inference_lite(x)
        p = self.nn(inp=out,ztemplates=self.Zs)
        # z = self.loaded_encoder(tf.expand_dims(x, axis=0))
        # p = self.dist_fn([self.Zs, z])
        return self.Xs[np.argmax(p)]


def generate_output(input_image, bbox_list, predictions, FPS):
    image2 = input_image.copy()
    draw = ImageDraw.Draw(image2)
    for bbox in bbox_list:
        draw.rectangle(bbox, outline="red", width=2)

    # Dimensions of the original image
    width, height = image2.size

    # Create a new image with increased width for the black rectangle
    new_width = width + 100  # Adjust the width of the black rectangle as needed
    new_image = Image.new('RGB', (new_width, height), color='black')

    # Paste the original image onto the left side of the new image
    new_image.paste(image2, (0, 0))

    # Draw a black rectangle on the right side
    draw = ImageDraw.Draw(new_image)
    draw.rectangle([(width, 0), (new_width, height)], fill="black")

    for i, bbox in enumerate(bbox_list):
        image_to_add = normalize(predictions[i]).numpy()
        image_to_add = Image.fromarray(np.uint8(255 * image_to_add))

        paste_position = ((width + new_width) // 2 - 32, (height // 5) + i * (height // 5))

        # Paste the image into the black region
        new_image.paste(image_to_add, paste_position)

    font = ImageFont.truetype("fonts/OpenSans-Bold.ttf", size=15)  # You can also load a specific font if needed
    text = f"FPS={round(FPS, 1)}"
    text_width = draw.textlength(text=text, font=font)
    text_position = (1010, 300)
    draw.text(text_position, text, fill="yellow", font=font)

    return new_image

def compute_FPS(read_input_time,inference_time):
    return 1 / (read_input_time + inference_time)

def generate_image_sequences(start_frame, end_frame, folder_name):
    images = []
    fps_list = []
    print("Inference proccedure is started")
    for i in tqdm(range(start_frame,end_frame)):
        s_time = time()
        img, bounding_box = read_input_frame(frame_idx=i,data=new_data,folder_name=folder_name)
        read_input_time = time() - s_time

        bbox_list = []
        for i in range(len(bounding_box[0])):
            bbox_list += [[bounding_box[0][i],bounding_box[1][i],bounding_box[2][i],bounding_box[3][i]]]

        s_time = time()
        predictions = []
        for bbox in bbox_list:
            extracted_image = tf.constant(extract_sign(img, bbox=bbox, target_size=(64,64)),dtype="float32")
            predictions += [infer(extracted_image)]
        inference_time = time() - s_time


        fps_list += [compute_FPS(read_input_time,inference_time)]
        #FPS = compute_FPS(read_input_time,inference_time)

        final_image = generate_output(input_image=img, bbox_list=bbox_list, predictions=predictions, FPS=np.mean(fps_list))

        images += [final_image]

    return images


def generate_video(scenario_dict):
    for sc in scenario_dict:
        print(f"Scenario: {sc['scenarioID']} is started")
        images = generate_image_sequences(start_frame=sc["startFrame"], end_frame=sc["startFrame"] + sc["numberFrame"],folder_name=sc["path"])
        video_file_name = "output_video_sc_" + str(sc["scenarioID"]) + ".avi"
        video_file = os.path.join(video_path,video_file_name)
        print(video_file)
        width, height = images[0].size
        video = cv2.VideoWriter(video_file, 0,10, (width,height))
        print("Video generation is started")
        for image in tqdm(images):
            img_np = np.array(image)
            frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            video.write(frame)

if __name__ == "__main__":
    labels_path = config["labels_path"]
    loader = get_loader(config["loader"])
    encoder_file = config["encoder_file"]
    video_path = config["video_path"]
    scenarios = config["scenarios"]
    bbox_convert_dict = config["bbox_convert_dict"]

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    data = pd.read_csv(labels_path, sep=";").astype(bbox_convert_dict)
    new_data = data.groupby("file_name").agg(list)
    new_data.reset_index(inplace=True)

    Xs = loader.get_test_generator_for_video()

    infer = Inference(encoder_h5=encoder_file, lite_encoder_file="model_files/best_encoders/new_2_student_gtsrb2tt100k_encoder_s1.tflite", template_set=Xs)

    generate_video(scenario_dict=scenarios)


