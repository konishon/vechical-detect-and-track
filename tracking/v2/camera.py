import cv2
class Camera():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        print("[info]Loading camera")

    def get_frame(self):
        s, img = self.cap.read()
        if s:
            pass

        return img

    def release_camera(self):
        self.cap.release()