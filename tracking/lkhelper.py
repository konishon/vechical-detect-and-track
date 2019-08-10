import cv2
import numpy as np
# lucas kanade params
feature_params = dict(maxCorners=100, qualityLevel=0.3,
                      minDistance=100, blockSize=17)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (100, 3))