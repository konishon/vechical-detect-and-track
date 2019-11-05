import cv2
import numpy as np


class OpticalFlowTracker:
    tracked_points = []

    def __init__(self):
        self.old_point = None
        self.old_gray = None
        self.x = None
        self.y = None

    def get_tracked_points(self):
        return self.tracked_points

    def start_track(self, rgb, points):
        self.old_point = np.array([[points[0], points[1]]], dtype=np.float32)
        self.old_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    def get_position(self):
        return self.x, self.y

    def update(self, rgb):
        new_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        lk_params = dict(winSize=(15, 15),
                         maxLevel=4,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        new_points, status, error = cv2.calcOpticalFlowPyrLK(self.old_gray, new_gray, self.old_point, None, **lk_params)
        self.x, self.y = new_points.ravel()

        self.old_gray = new_gray.copy()
        self.old_point = new_points.reshape(-1, 1, 2)
