import cv2
import fire
import imutils
from tracking.mathhelper import get_bb_from_centroid
from tracking.v2.optical_flow_tracker import OpticalFlowTracker
from tracking.v2.py.centroidtracker import CentroidTracker
from tracking.v2.py.trackableobject import TrackableObject
from tracking.v2.retina_net_camera import RetinaNetCamera

trackable_objects = {}


def run(src):
    frame_number = 0
    trackers = []
    points = []
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)

    retina_camera = RetinaNetCamera(src)
    while True:
        has_preds, frame, preds = retina_camera.get_frame(frame_number)

        if has_preds:
            boxes, scores, labels, = preds
            for i, (box, score, label) in enumerate(zip(boxes[0], scores[0], labels[0])):
                tracker = OpticalFlowTracker()
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                tracker.start_track(frame, [center_x, center_y])
                trackers.append(tracker)
        else:
            print("Tracking {} objects".format(len(trackers)))
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

                # add the bounding box coordinates to the rectangles list
                points.append((start_x, start_y, end_x, end_y))

        objects = ct.update(points)
        for (objectID, centroid) in objects.items():
            to = trackable_objects.get(objectID, None)
            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                to.centroids.append(centroid)

            trackable_objects[objectID] = to

            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        frame = imutils.resize(frame, width=500)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(5) & 0xFF

        if key == 27:
            retina_camera.release_camera()
            break

        frame_number = frame_number + 1


if __name__ == '__main__':
    fire.Fire(run)
