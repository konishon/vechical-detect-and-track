# retinanet imports [starts]
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import os
import tensorflow as tf
import numpy as np
# retinanet imports [ends]

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def load_model():
    # set the modified tf session as backend in keras
    # keras.backend.tensorflow_backend.set_session(get_session())

    # # load retinanet model
    # model_path = os.path.join('snapshots', 'resnet50_csv_12_inference.h5')
    # print(model_path)
    # model = models.load_model(model_path, backbone_name='resnet50')
    return model
