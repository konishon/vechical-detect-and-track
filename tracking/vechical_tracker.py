import argparse
import os

import cv2
import fire
import dlib

import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from v2.retina_net_camera import RetinaNetCamera
from v2.optical_flow_tracker import OpticalFlowTracker



def run(src):
    frame_number = 0
    retina_camera = RetinaNetCamera(src)
    while True:
      has_preds,frame,preds = retina_camera.get_frame(frame_number)
    
      if has_preds:
        boxes, scores, labels, = preds
        for i, (box, score, label) in enumerate(zip(boxes[0], scores[0], labels[0])):
          tracker = OpticalFlowTracker()
          centerX = (box[0] + box[2]) / 2
          centerY = (box[1] + box[3]) / 2
          point = np.array([[centerX, centerY]], dtype=np.float32)
          tracker.start_track(frame, [centerX, centerY])

      cv2.imshow("Frame", frame)
      key = cv2.waitKey(5) & 0xFF

      if key == 27:
        retina_camera.release_camera()
        break

      frame_number = frame_number + 1

if __name__ == '__main__':
  fire.Fire(run)