import cv2
import fire
import imutils
from tracking.mathhelper import get_bb_from_centroid
from tracking.v2.optical_flow_tracker import OpticalFlowTracker
from tracking.v2.iou_tracker import IOUTracker
from tracking.v2.py.trackableobject import TrackableObject
from tracking.v2.retina_net_camera import RetinaNetCamera

trackable_objects = {}

labels_to_names = {0: 'Biker',
                   1: 'Car',
                   2: 'Bus',
                   3: 'Cart',
                   4: 'Skater',
                   5: 'Pedestrian'}


def create_new_trackers(preds, frame, frame_number, trackers):
    boxes, scores, labels, = preds
    for i, (box, score, label) in enumerate(zip(boxes[0], scores[0], labels[0])):
        if(score < 0.40):
            continue

        tracker = OpticalFlowTracker()
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        label = "{}_{}_{}".format(labels_to_names[label], frame_number, i)

        tracker.start_track(frame, [center_x, center_y], label)
        trackers.append(tracker)


def update_position(frame,trackers):
    # print("Tracking {} objects".format(len(trackers)))
    for tracker in trackers:
        tracker.update(frame)
        (x, y) = tracker.get_position()

        x = int(x)
        y = int(y)

        (start_x, start_y), (end_x, end_y) = get_bb_from_centroid(x, y, 20, 20)

        # unpack the position object
        start_x = int(start_x)
        start_y = int(start_y)
        end_x = int(end_x)
        end_y = int(end_y)

        text = "ID {}".format(tracker.label)
        cv2.putText(frame, text, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


def run(src):
    frame_number = 0
    trackers = []
    points = []
    ct = IOUTracker()

    retina_camera = RetinaNetCamera(src)
    while True:
        has_preds, frame, preds = retina_camera.get_frame(frame_number)

        if has_preds:
            create_new_trackers(preds, frame, frame_number, trackers)
        else:
            update_position(frame,trackers)

        frame = imutils.resize(frame, width=500)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            retina_camera.release_camera()
            break

        frame_number = frame_number + 1

if __name__ == '__main__':
    fire.Fire(run)
