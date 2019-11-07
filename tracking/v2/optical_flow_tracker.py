import cv2
import numpy as np
import time


class OpticalFlowTracker:
    tracked_points = []

    def __init__(self):
        self.old_point = None
        self.old_gray = None
        self.x = None
        self.y = None
        self.label = None

    def crop_image(self, img, bb, padding=50):
        (x1, y1, x2, y2) = bb

        crop_img = img[int(y1):int(y2), int(x1):int(x2)]
        # image_name = "{0}.jpg".format(time.time())
        # cv2.imwrite(image_name, crop_img)

        return crop_img

    '''
    Takes a grayscale frame and boundingbox
    Crops the frame using the bounding box
    Runs corner detection on the cropped image
    '''

    def corners_to_track(self, gray_frame, bb):
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)

        cropped_img = self.crop_image(gray_frame, bb)
        (x1, y1, x2, y2) = bb
        scaled_corners = []
        corners = cv2.goodFeaturesToTrack(
            cropped_img, mask=None, **feature_params)
        if corners is None:
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            scaled_corners.append([center_x, center_y])
        else:
            # cv2.imshow("crop_{}".format(bb), cropped_img)

            for point in corners:
                x = point[0][0] + x1
                y = point[0][1] + y1
            scaled_corners.append([x, y])

        return scaled_corners

    def get_tracked_points(self):
        return self.tracked_points

    '''
    Takes a RGB frame, bounding boxes and labels 
    then uses corner detection to on the cropped images 
    to find good points for optical flow
    '''

    def start_track(self, rgb, bb, label):

        self.label = label

        self.old_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        self.old_point = np.array([[0, 0]], dtype=np.float32)

        corners = self.corners_to_track(self.old_gray, bb)

        if corners is not None:
            for point in corners:

                x, y = point[0], point[1]
                point_to_concat = np.array([[x, y]], dtype=np.float32)
                self.old_point = np.concatenate(
                    [self.old_point, point_to_concat])

            if len(self.old_point) > 1:
                self.old_point = np.delete(self.old_point, 0, axis=0)

    def get_position(self):
        return self.new_points

    '''
    Takes a RGB 
    Then makes it Grayscale
    Calculates Optical flow
    '''

    def update(self, rgb):
        new_gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        lk_params = dict(winSize=(15, 15),
                         maxLevel=4,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        new_points_all, status, error = cv2.calcOpticalFlowPyrLK(
            self.old_gray, new_gray, self.old_point, None, **lk_params)
        self.new_points = new_points_all.ravel()

        self.old_gray = new_gray.copy()
        self.old_point = new_points_all.reshape(-1, 1, 2)
