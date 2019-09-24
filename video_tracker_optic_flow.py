import argparse
import os
from collections import deque
from itertools import islice

import cv2
import imutils
import numpy as np

from aerial_pedestrian_detection.deep_sort.deep_sort.detection import Detection
from aerial_pedestrian_detection.deep_sort.application_util import preprocessing
from tracking.deep_sort_tracker import DeepSort
from tracking.lkhelper import lk_params, color
from tracking.mathhelper import get_bb_from_centroid, iou_from_shapely
from tracking.retinaNet import retinaNet
from tracking.retinahelper import labels_to_names
from tracking.trackableobject import TrackableObject

stream = None
skip_frames = 0
current_frame_number = 0
trackable_objects = {}
show_preview = False



def capture_args():
    global stream, skip_frames, show_preview
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True, type=str, help="path to input video file")
    ap.add_argument("-s", "--skip_frames", required=True, type=int,
                    help="runs object detector every specified n frames")
    ap.add_argument("-p", "--show_preview", default=False, required=False, type=bool,
                    help="should show preview?")
    args = vars(ap.parse_args())

    stream = cv2.VideoCapture(args['video'])
    skip_frames = args['skip_frames']
    show_preview = args['show_preview']


def image_fprop(net, image):
    if type(image) == str:
        image = cv2.imread(image)
    detections = net.forward(image)
    return detections


def append_to_tracker(detections):
    found_objects_centroids = []
    for i, detection in enumerate(detections):
        object_id = i + len(trackable_objects)
        box = detection['bbox']
        label = detection['label']

        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        to = TrackableObject(object_id, (center_x, center_y),
                             deque(), label)
        trackable_objects[object_id] = to
        found_objects_centroids.append(center_x)
        found_objects_centroids.append(center_y)
    return found_objects_centroids


def chunk(it, size):
    it = iter(it)
    return list(iter(lambda: list(islice(it, size)), []))


def predict_on_video(net, tracker_wrapper):
    global current_frame_number

    while True:
        ret, frame = stream.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if current_frame_number % skip_frames == 0:
            detections = image_fprop(net, frame)
            num_detections = int(len(detections))
            print("[Info] found {0} objects at frame {1}".format(num_detections, current_frame_number))
            found_objects_centroids = append_to_tracker(detections)

        if show_preview:
            resized = imutils.resize(image, width=1000)
            cv2.imshow('frame', resized)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        # increment frame count
        current_frame_number = current_frame_number + 1


def map_prediction_to_tracker(boxes, scores, labels):
    for i, (box, score, label) in enumerate(zip(boxes[0], scores[0], labels[0])):
        if score < 0.6:
            continue

        object_id = i + len(trackable_objects)

        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        to = TrackableObject(object_id, (center_x, center_y),
                             deque(), labels_to_names[label])
        trackable_objects[object_id] = to


if __name__ == '__main__':
    model_path = os.path.join('snapshots', 'resnet50_csv_08_inference.h5')
    deep_sort_model_path = "aerial_pedestrian_detection/deep_sort/resources/networks/mars-small128.pb"
    retinaNet = retinaNet(model_path, 0.6)
    t = DeepSort(deep_sort_model_path, max_age=1, max_distance=0.5, nn_budget=150,
                 nms_max_overlap=1.0, n_init=1)
    capture_args()

    predict_on_video(retinaNet, None)
