import cv2
import fire
import imutils

from tracking.v2.optical_flow_tracker import OpticalFlowTracker


from tracking.v2.retina_net_camera import RetinaNetCamera
import numpy as np

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
        if(score < 0.50):
            continue
        if label == 5:
            continue
    

        tracker = OpticalFlowTracker()
        # label = "{}_{}_{}".format(labels_to_names[label], frame_number, i)
        label = "{}".format(labels_to_names[label])
        color = get_random_color()

        tracker.start_track(frame, box, label)
        tracker.assign_color(color)
        trackers.append(tracker)

def is_the_detection_new(box,trackers):
    is_cointained = []
    for tracker in trackers:
        points = tracker.get_position()


        
def rectContains(rect,pt):
    logic = rect[0] < pt[0] < rect[0]+rect[2] and rect[1] < pt[1] < rect[1]+rect[3]
    return logic

def get_random_color():
    color1 = (list(np.random.choice(range(256), size=3)))
    color = (int(color1[0]), int(color1[1]), int(color1[2]))
    return color


def update_position(frame, trackers):
    # print("Tracking {} objects".format(len(trackers)))
    for tracker in trackers:
        tracker.update(frame)
        points = tracker.get_position()

        points = chunker(points, 2)

        for point in points:

            x = int(point[0])
            y = int(point[1])

            text = "ID {}".format(tracker.label)
            cv2.putText(frame, text, (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, tracker.color, 2)
            cv2.circle(frame, (x, y), 1, tracker.color, -1)


def chunker(seq, size):
    res = []
    for el in seq:
        res.append(el)
        if len(res) == size:
            yield res
            res = []
    if res:
        yield res



def run(src):
    frame_number = 0
    trackers = []
    points = []

    # # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # # We convert the resolutions from float to integer.
    frame_width = int(800)
    frame_height = int(800)

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('optical-flow.avi', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    retina_camera = RetinaNetCamera(src)
    while True:
        has_preds, frame, preds = retina_camera.get_frame(frame_number)

        if has_preds:
            create_new_trackers(preds, frame, frame_number, trackers)
        else:
            update_position(frame, trackers)

        frame = imutils.resize(frame, width=800)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            retina_camera.release_camera()
            break

        frame_number = frame_number + 1


if __name__ == '__main__':
    fire.Fire(run)
