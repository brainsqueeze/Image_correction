#  __author__ = 'Dave'
import cv2
from skimage import io
import os


class CorrectImage(object):

    def __init__(self, image_path):
        self.pic = image_path
        # self.img = self._load_image()

    def load_image(self, image):
        """
        :param image: image file name (str)
        :return: skimage image data
        """
        filename = os.path.join(self.pic, image)
        return io.imread(filename)

    def detect_edges(self, image):
        """
        :param image: image file name (str)
        :return: detected edges with variable filters
        """
        def nothing(x):
            pass

        img = self.load_image(image)
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
