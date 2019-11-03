import cv2
import numpy as np
from aerial_pedestrian_detection.keras_retinanet.utils.image import preprocess_image
from tracking.retinahelper import load_model

'''
Takes model and video and outputs detections for each frame
'''


class RetinaNetCamera:
    def __init__(self, video_source, show_bb=True):
        self.cap = cv2.VideoCapture(video_source)
        self.model = load_model()
        self.show_bb = show_bb

    def get_frame(self):
        s, img = self.cap.read()
        if s:
            pass
        img, bb = self.forward_pass(img)
        return img, bb

    def forward_pass(self, frame):
        frame = preprocess_image(frame)
        boxes, scores, labels = self.model.predict_on_batch(
            np.expand_dims(frame, axis=0))

        if self.show_bb:
            self.draw_bb(boxes, frame)

        return frame, (boxes, scores, labels)

    @staticmethod
    def draw_bb(faces, img):
        padding = 10

        for (x, y, w, h) in faces:
            x1 = x - padding
            y1 = y - padding
            x2 = x + w + padding
            y2 = y + h + padding

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return img
