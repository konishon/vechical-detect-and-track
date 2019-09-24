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
from tracking.mathhelper import get_bb_from_centroid, iou_from_shapely, chunk
from tracking.retinaNet import retinaNet
from tracking.retinahelper import labels_to_names
from tracking.trackableobject import TrackableObject

stream = None
skip_frames = 0
current_frame_number = 0
trackable_objects = {}
show_preview = False

# optic flow
old_gray = None


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


def predict_on_video(net):
    global current_frame_number, old_gray
    found_objects_centroids = []
    while True:
        ret, frame = stream.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(found_objects_centroids) > 0:
            good_old = chunk(found_objects_centroids, 2)
            good_new, st, err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, good_old, None, **lk_params)

            match_old_to_new_point(frame, good_new, good_old)

        if current_frame_number % skip_frames == 0:
            detections = image_fprop(net, frame)
            num_detections = int(len(detections))
            print("[Info] found {0} objects at frame {1}".format(num_detections, current_frame_number))
            found_objects_centroids = append_to_tracker(detections)

        if show_preview:
            resized = imutils.resize(frame, width=300)
            cv2.imshow('frame', resized)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        old_gray = frame_gray.copy()
        # increment frame count
        current_frame_number = current_frame_number + 1


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


def match_old_to_new_point(frame, good_new, good_old):
    for i, new_point in enumerate(good_new):
        a, b = new_point.ravel()
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), 20)

        top_left_new, bottom_right_new = get_bb_from_centroid(
            a, b, 30, 30)

        x1_top_left = top_left_new[0]
        y1_top_left = top_left_new[1]

        x2_bottom_right = bottom_right_new[0]
        y2_bottom_right = bottom_right_new[1]

        new_bb = {'x1': x1_top_left,
                  'x2': x2_bottom_right,
                  'y1': y1_top_left,
                  'y2': y2_bottom_right}

        # new_bb = [x1_top_left, y1_top_left, x2_bottom_right, y2_bottom_right]

        cv2.rectangle(frame, (int(x1_top_left), int(y1_top_left)),
                      (int(x2_bottom_right), int(y2_bottom_right)), (0, 255, 0), 2)

        for old_point in good_old:
            c, d = old_point.ravel()
            top_left_old, bottom_right_old = get_bb_from_centroid(
                c, d, 30, 30)

            x1_top_left_old = top_left_old[0]
            y1_top_left_old = top_left_old[1]

            x2_bottom_right_old = bottom_right_old[0]
            y2_bottom_right_old = bottom_right_old[1]

            old_bb = {'x1': x1_top_left_old,
                      'x2': x2_bottom_right_old,
                      'y1': y1_top_left_old,
                      'y2': y2_bottom_right_old}

            # old_bb = [x1_top_left_old, y1_top_left_old,
            #           x2_bottom_right_old, y2_bottom_right_old]

            cv2.rectangle(frame, (int(x1_top_left_old), int(y1_top_left_old)),
                          (int(x2_bottom_right_old), int(y2_bottom_right_old)), (255, 0, 0), 2)

            overlap_ratio = iou_from_shapely((c, d), (a, b))
            if overlap_ratio > 0.90:
                # print("Overlap ratio of {0} with {1} with centroid {2} ====> {3}".format(
                #     "", "", old_point, overlap_ratio))

                key_to_update = None
                for key, value in trackable_objects.items():
                    cur_old_point = (old_point.ravel()[
                                         0], old_point.ravel()[1])
                    if cur_old_point == value.centroids:
                        key_to_update = key

                if key_to_update is not None:
                    x, y = new_point.ravel()
                    print("Found for {0}".format(new_point.ravel()))
                    point_to_update = (x, y)
                    found_obj = trackable_objects[key_to_update]
                    found_obj.centroids = point_to_update
                    # print("Updated {0}".format(trackableObjects))

                    text = "{0} {1}".format(found_obj.label, found_obj.objectID)
                    draw_text(frame, found_obj.centroids, text)
                else:
                    pass
                    # print("Searching for {0}".format(new_point.ravel()))
                    # print("In {0}".format(trackableObjects.items()))
                    # raise ValueError("We did not find a matching key")


def draw_text(frame, coords, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, str(text), coords, font,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)


if __name__ == '__main__':
    model_path = os.path.join('snapshots', 'resnet50_csv_08_inference.h5')
    capture_args()
    retinaNet = retinaNet(model_path, 0.6)
    predict_on_video(retinaNet)
