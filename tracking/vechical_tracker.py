import argparse
import os

import cv2
import fire

import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from v2.retina_net_camera import RetinaNetCamera



def run(src):
    frame_number = 0
    retina_camera = RetinaNetCamera(src)
    while True:
      frame,pred = retina_camera.get_frame(frame_number)
      cv2.imshow("Frame", frame)
      
      key = cv2.waitKey(5) & 0xFF

      if key == 27:
        retina_camera.release_camera()
        break

      frame_number = frame_number + 1

if __name__ == '__main__':
  fire.Fire(run)