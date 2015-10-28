#  __author__ = 'Dave'
import cv2
from skimage import io
from skimage.transform import hough_transform, probabilistic_hough_line

import matplotlib.pyplot as plt

import os


class CorrectImage(object):

    def __init__(self, image_path):
        self.path = image_path

    def load_image(self, image):
        """
        :param image: image file name (str)
        :return: skimage image data
        """
        filename = os.path.join(self.path, image)
        return io.imread(filename)

    def detect_edges(self, image, vary=False, plot=False):
        """
        :param image: image file name (str)
        :param vary: turn tunable plotting on
        :param plot: turn plotting on
        :return: detected edges with variable filters
        """
        img = self.load_image(image)
        if vary:
            def nothing(x):
                pass

            cv2.namedWindow('image')
            cv2.createTrackbar('th1', 'image', 0, 255, nothing)
            cv2.createTrackbar('th2', 'image', 0, 255, nothing)

            while True:
                th1 = cv2.getTrackbarPos('th1', 'image')
                th2 = cv2.getTrackbarPos('th2', 'image')
                edges = cv2.Canny(img, th1, th2)
                cv2.imshow('image', edges)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break

            cv2.destroyAllWindows()
        edges = cv2.Canny(img, 255, 255)
        if plot:
            cv2.namedWindow('image')
            cv2.imshow('image', edges)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()
        return edges

    def hough_transform(self, image, plot=False):
        """
        :param image: image file name (str)
        :param plot: turn plotting on
        :return: numpy array of probabilistically found straight lines
        """
        edges = self.detect_edges(image)
        lines = probabilistic_hough_line(edges, threshold=10, line_length=5, line_gap=3)
        if plot:
            for line in lines:
                p0, p1 = line
                plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
            plt.show()
        return lines
