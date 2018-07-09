#  __author__ = 'Dave'
import cv2
from skimage import io
from skimage.transform import probabilistic_hough_line

import matplotlib.pyplot as plt
import os
import warnings
import random
import numpy as np

warnings.filterwarnings('ignore', category=RuntimeWarning)


class CorrectImage(object):

    def __init__(self):
        self.path = ""
        self.name = ""
        self.image = None
        self.edges = None
        self.lines = None

    def _load_image(self, image):
        """
        :param image: image file name (str)
        :return: skimage image data
        """
        filename = os.path.join(self.path, image)
        return io.imread(filename)

    def add_path(self, image_path):
        """
        Adds image to the list of images
        :param image_path: (string)
        """
        self.path = image_path + '/'

    def add_image(self, filename):
        """
        Adds image to the list of images
        :param filename: (string)
        """
        self.name = filename
        self.hough_transform()

    def _detect_edges(self, image, vary=False, plot=False):
        """
        :param image: image file name (str)
        :param vary: turn tunable plotting on
        :param plot: turn plotting on
        :return: detected edges with variable filters
        """
        self.image = self._load_image(image)
        if vary:
            def nothing(x):
                pass

            cv2.namedWindow('image')
            cv2.createTrackbar('th1', 'image', 0, 255, nothing)
            cv2.createTrackbar('th2', 'image', 0, 255, nothing)

            while True:
                th1 = cv2.getTrackbarPos('th1', 'image')
                th2 = cv2.getTrackbarPos('th2', 'image')
                edges = cv2.Canny(self.image, th1, th2)
                cv2.imshow('image', edges)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break

            cv2.destroyAllWindows()
        edges = cv2.Canny(self.image, 255, 255)
        if plot:
            cv2.namedWindow('image')
            cv2.imshow('image', edges)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()
        return edges

    def hough_transform(self, vary=False, plot=False):
        """
        :param vary: turn edge detection tunable plotting on
        :param plot: turn plotting on
        :return: numpy array of probabilistically found straight lines
        """
        if self.name == "":
            raise ValueError('Missing image: you need to specify the image file using add_image.')

        self.edges = self._detect_edges(self.name, vary=vary, plot=plot)
        self.lines = probabilistic_hough_line(self.edges, threshold=10, line_length=5, line_gap=3)
        if plot:
            for line in self.lines:
                p0, p1 = line
                plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
            plt.show()

    @staticmethod
    def slope(lines):
        """
        :param lines: array of coordinates (ie. [((x0, y0), (xf, yf)), ...]
        :return: array of slope values with the same number of entries as lines
        """

        # for doing vectorized subtraction across all line pairs,
        # we need the first line of each pair to be the negative of itself
        sign_op = np.ones_like(lines)
        sign_op[:, :, 0] *= -1

        # get the differences between x and y coordinates (start, end), respectively
        slopes = np.sum(sign_op * lines, axis=2)

        # compute the slopes of each line for every line pair
        slopes = np.divide(slopes[:, :, 0], slopes[:, :, 1])

        # turn infinite values to a finite, but very large value
        slopes[np.isinf(slopes)] = 1e6

        return slopes

    def line_pair(self, num_pairs):
        """
        :param num_pairs: number of line pairs to take (int)
        :return: line pairs (array)
        """

        idx = np.random.randint(len(self.lines), size=num_pairs * 2)
        lines = np.array(self.lines)[idx]
        return lines.reshape(num_pairs, 2, 2, 2)

    @staticmethod
    def mutation(pairs, p_mutate=0.01):
        """
        :param pairs: (numpy array with dimensions (n_pairs, 2, 2, 2)) pairs of lines
        :param p_mutate: (float) probability of a mutation
        :return: (numpy array with dimensions (n_pairs, 2, 2, 2)) pairs of lines with mutations
        """

        for i in range(len(pairs)):
            if p_mutate > random.random():
                # column = np.random.randint(low=0, high=2)
                for column in [0, 1]:
                    t = pairs[i, :, :, column]
                    low, high = np.min(t), np.max(t)
                    if high == low:
                        high *= 2
                    pairs[i, :, :, column] = np.random.randint(low=low, high=high, size=t.shape)
        return pairs
