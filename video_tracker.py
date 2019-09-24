import argparse
import os
from collections import deque
import cv2
import imutils
import numpy as np

from aerial_pedestrian_detection.deep_sort.deep_sort.detection import Detection
from aerial_pedestrian_detection.deep_sort.application_util import preprocessing
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


def predict_on_video(net, tracker_wrapper):
    global current_frame_number
    tracker = tracker_wrapper.tracker
    while True:
        ret, frame = stream.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if current_frame_number % skip_frames == 0:
            detections = image_fprop(net, frame)
            num_detections = int(len(detections))
            print("[Info] found {0} objects at frame {1}".format(num_detections, current_frame_number))
            boxes = []

            for det in detections:
                boxes.append(det['bbox'])

            scaled_boxes = get_xywh(boxes)
            features = tracker_wrapper.encoder(image, scaled_boxes)

            # using deep sort to track
            # score to 1.0 here).
            tr_detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(scaled_boxes, features)]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in tr_detections])
            scores = np.array([d.confidence for d in tr_detections])
            indices = preprocessing.non_max_suppression(boxes, tracker_wrapper.nms_max_overlap, scores)
            tr_detections = [tr_detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(tr_detections)
        else:
            tracker.predict()

        print(len(tracker.tracks))
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(image, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)

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
    retinaNet = retinaNet(model_path, 0.6)
    t = DeepSort(deep_sort_model_path, max_age=1, max_distance=0.5, nn_budget=150,
                       nms_max_overlap=1.0, n_init=1)
    capture_args()
    predict_on_video(retinaNet, t)
