import cv2
import imutils
import numpy as np
import keras
from keras_retinanet import models
import tensorflow as tf

'''
Takes model and video and outputs detections for each frame
'''


class RetinaNetCamera:
    def __init__(self, video_source, show_bb=True, export_video=False):
        self.cap = cv2.VideoCapture(video_source)
        self.model = self.load_model()
        self.show_bb = show_bb
        self.current_frame = None
        if export_video:
            self.init_video_export()

    def release_camera(self):
        self.cap.release()

    def update_current_frame(self, current_frame):
        self.current_frame = current_frame

    def get_frame(self, frame_number):
        self.update_current_frame(frame_number)
        s, img = self.cap.read()
        frame = imutils.resize(img, width=800)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        bb = None
        has_preds = 0
        if s:
            pass
        if frame_number % 100 == 0:
            has_preds = 1
            img, bb = self.forward_pass(rgb)
        return has_preds, rgb, bb

    def forward_pass(self, frame):
        frame = self.preprocess_image(frame)
        boxes, scores, labels = self.model.predict_on_batch(
            np.expand_dims(frame, axis=0))

        if self.show_bb:
            pass

        return frame, (boxes, scores, labels)

    def preprocess_image(self, x, mode='caffe'):
        """ Preprocess an image by subtracting the ImageNet mean.

        Args
            x: np.array of shape (None, None, 3) or (3, None, None).
            mode: One of "caffe" or "tf".
                - caffe: will zero-center each color channel with
                    respect to the ImageNet dataset, without scaling.
                - tf: will scale pixels between -1 and 1, sample-wise.

        Returns
            The input with the ImageNet mean subtracted.
        """
        # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
        # except for converting RGB -> BGR since we assume BGR already

        # covert always to float32 to keep compatibility with opencv
        x = x.astype(np.float32)

        if mode == 'tf':
            x /= 127.5
            x -= 1.
        elif mode == 'caffe':
            x[..., 0] -= 103.939
            x[..., 1] -= 116.779
            x[..., 2] -= 123.68

        return x

    def init_video_export(self):
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))

        self.out = cv2.VideoWriter('optical-flow.avi', cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    def add_frame(self, frame):
        if self.out is not None:
            self.out.write(frame)

    @staticmethod
    def get_session():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def load_model(self):
        # set the modified tf session as backend in keras
        keras.backend.tensorflow_backend.set_session(self.get_session())

        # # load retinanet model
        # model_path = os.path.join('snapshots', 'resnet50_csv_12_inference.h5')
        # print(model_path)
        model = models.load_model(
            "/home/nishon/Projects/python/vechical-detect-and-track/snapshots/resnet50_csv_12_inference.h5",
            backbone_name='resnet50')
        return model
