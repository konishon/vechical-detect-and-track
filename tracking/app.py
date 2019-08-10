import argparse
from collections import deque
import cv2
import imutils

import numpy as np

from trackableobject import TrackableObject
from lkhelper import color, lk_params
from mathhelper import get_iou, get_bb_from_centroid, bb_intersection_over_union, bb_intersection_over_union2, \
    iou_from_shapely
from retinahelper import load_model, get_good_features_to_track, labels_to_names

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="path to input video file")
ap.add_argument("-s", "--skip_frames", type=int,
                help="runs object dectector every specified n frames")
args = vars(ap.parse_args())
stream = cv2.VideoCapture(args['video'])
skip_frames = args['skip_frames']
current_frame_number = 1

cap = cv2.VideoCapture(args["video"])
abpoints = {}
removeBadValue = True
trackableObjects = {}
model = load_model()

# loading intital first points
ret, old_frame = cap.read()
p0 = np.array([[0, 0]], dtype=np.float32)
boxes, scores, labels, = get_good_features_to_track(old_frame, model)
for i, (box, score, label) in enumerate(zip(boxes[0], scores[0], labels[0])):
    if score < 0.6:
        continue

    if label == 5:
        continue

    b = box.astype(int)
    centerX = (box[0] + box[2]) / 2
    centerY = (box[1] + box[3]) / 2
    point = np.array([[centerX, centerY]], dtype=np.float32)
    p0 = np.concatenate((point, p0))

    objectID = i
    to = TrackableObject(objectID, (centerX, centerY),
                         deque(), labels_to_names[label])
    trackableObjects[objectID] = to


    if removeBadValue:
        p0 = np.delete(p0, (1), axis=0)
        removeBadValue = False

print("Added {0}".format(trackableObjects))
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(old_frame)


def draw_text(frame, coords, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, str(text), coords, font,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)


# # Default resolutions of the frame are obtained.The default resolutions are system dependent.
# # We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('unique-names.avi', cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 10, (frame_width, frame_height))


def attach_direction_tracker(x, y):
    if i not in abpoints.keys():
        abpoints[i] = [(x, y)]
        abpoints[i] = deque()
    else:
        abpoints[i].appendleft((x, y))

    if len(abpoints[i]) > 10:
        # compute the difference between the x and y
        # coordinates and re-initialize the direction
        # text variables
        dX = abpoints[i][-10][0] - abpoints[i][0][0]
        dY = abpoints[i][-10][1] - abpoints[i][0][1]
        (dirX, dirY) = ("", "")

        # ensure there is significant movement in the
        # x-direction
        if np.abs(dX) > 20:
            dirX = "East" if np.sign(dX) == 1 else "West"

        # ensure there is significant movement in the
        # y-direction
        if np.abs(dY) > 20:
            dirY = "North" if np.sign(dY) == 1 else "South"

        # handle when both directions are non-empty
        if dirX != "" and dirY != "":
            direction = "{}-{}".format(dirY, dirX)

        # otherwise, only one direction is non-empty
        else:
            direction = dirX if dirX != "" else dirY

        x = abpoints[i][0][0]
        y = abpoints[i][0][1]
        cv2.putText(frame, direction, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 3)


while 1:

    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        old_gray, frame_gray, p0, None, **lk_params)
    good_new = p1
    good_old = p0
 

    if current_frame_number % skip_frames == 0:
        print("Detecting Objects on {0}".format(current_frame_number))
        print("Found {0} with confidence {1}".format(labels_to_names[label],score))    
        boxes, scores, labels, = get_good_features_to_track(old_frame, model)
        for i, (box, score, label) in enumerate(zip(boxes[0], scores[0], labels[0])):
            # if score < 0.2:
            #     continue

            if label == 5:
                continue

            b = box.astype(int)
            centerX = (box[0] + box[2]) / 2
            centerY = (box[1] + box[3]) / 2
            point = np.array([[centerX, centerY]], dtype=np.float32)            
            p0 = np.append(p0,point)

    
            objectID = i + len(trackableObjects)
            to = TrackableObject(objectID, (centerX, centerY),
                                deque(), labels_to_names[label])
            trackableObjects[objectID] = to
             

    for i, new_point in enumerate(good_new):
        a, b = new_point.ravel()
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), 20)
        attach_direction_tracker(a, b)

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
                for key, value in trackableObjects.items():
                    cur_old_point = (old_point.ravel()[
                                     0], old_point.ravel()[1])
                    if cur_old_point == value.centroids:
                        key_to_update = key

                if key_to_update is not None:
                    x, y = new_point.ravel()
                    # print("Found for {0}".format(new_point.ravel()))
                    point_to_update = (x, y)
                    foundObj = trackableObjects[key_to_update]
                    foundObj.centroids = point_to_update
                    # print("Updated {0}".format(trackableObjects))

                    text = "{0} {1}".format(foundObj.label, foundObj.objectID)
                    draw_text(frame, foundObj.centroids, text)
                else:
                    pass
                    # print("Searching for {0}".format(new_point.ravel()))
                    # print("In {0}".format(trackableObjects.items()))
                    # raise ValueError("We did not find a matching key")

    img = cv2.add(frame, mask)
    out.write(img)
    resized = imutils.resize(img, width=1000)
    cv2.imshow('frame', resized)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    current_frame_number = current_frame_number + 1

cv2.destroyAllWindows()
cap.release()
