import argparse
import os
from collections import deque

import cv2
import imutils

from tracking.deep_sort_tracker import DeepSort
from tracking.retinaNet import retinaNet
from tracking.retinahelper import labels_to_names
from tracking.trackableobject import TrackableObject

stream = None
skip_frames = 0
current_frame_number = 0
trackable_objects = {}



def capture_args():
    global stream, skip_frames
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True, type=str, help="path to input video file")
    ap.add_argument("-s", "--skip_frames", required=True, type=int,
                    help="runs object detector every specified n frames")
    args = vars(ap.parse_args())

    stream = cv2.VideoCapture(args['video'])
    skip_frames = args['skip_frames']


def image_fprop(net, image):
    if type(image) == str:
        image = cv2.imread(image)
    detections = net.forward(image)
    return detections


def predict_on_video(net,tracker_wrapper):
    global current_frame_number
    tracker = tracker_wrapper.tracker
    while True:
        ret, frame = stream.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if current_frame_number % skip_frames == 0:
            detections = image_fprop(net, frame)
            num_detections = int(len(detections))
            print("[Info] found {0} objects at frame {1}".format(num_detections, current_frame_number))

            boxes = []
            for det in detections:
                boxes.append(det['bbox'])

            scaled_boxes = get_xywh(boxes)
            features = tracker_wrapper.encoder(image,scaled_boxes)


        resized = imutils.resize(image, width=100)
        # cv2.imshow('frame', resized)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        # increment frame count
        current_frame_number = current_frame_number + 1


def map_prediction_to_tracker(boxes, scores, labels):
    for i, (box, score, label) in enumerate(zip(boxes[0], scores[0], labels[0])):
        if score < 0.6:
            continue

        object_id = i

        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        to = TrackableObject(object_id, (center_x, center_y),
                             deque(), labels_to_names[label])
        trackable_objects[object_id] = to


def get_xywh(boxes):
    ret_boxes = []
    for box in boxes:
        # box = box.astype("int")
        x = int(box[1])
        y = int(box[0])
        w = int(box[3] - box[1])
        h = int(box[2] - box[0])

        if x < 0:
            w = w + x
            x = 0
        if y < 0:
            h = h + y
            y = 0
        ret_boxes.append([x, y, w, h])
    return ret_boxes


if __name__ == '__main__':
    model_path = os.path.join('snapshots', 'resnet50_csv_08_inference.h5')
    deep_sort_model_path = "aerial_pedestrian_detection/deep_sort/resources/networks/mars-small128.pb"
    net = retinaNet(model_path, 0.6)
    tracker = DeepSort(deep_sort_model_path, max_age=1, max_distance=0.5, nn_budget=150,
                       nms_max_overlap=1.0, n_init=1)
    capture_args()
    predict_on_video(net,tracker)
