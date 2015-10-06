#  __author__ = 'Dave'
import numpy as np
import cv2
from skimage import io
import os


class CorrectImage(object):

    def __init__(self, image_path):
        self.pic = image_path
        self.img = self._load_image()

    def _load_image(self):
        filename = os.path.join(self.pic, 'initial.png')
        return io.imread(filename)

    def detect_edges(self):
        def nothing(x):
            pass

        cv2.namedWindow('image')
        cv2.createTrackbar('th1', 'image', 0, 255, nothing)
        cv2.createTrackbar('th2', 'image', 0, 255, nothing)

        while(1):
            th1 = cv2.getTrackbarPos('th1', 'image')
            th2 = cv2.getTrackbarPos('th2', 'image')
            edges = cv2.Canny(self.img, th1, th2)
            cv2.imshow('image', edges)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        cv2.destroyAllWindows()
