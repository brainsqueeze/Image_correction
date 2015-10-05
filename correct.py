#  __author__ = 'Dave'
import numpy as np
import cv2
from skimage import io
import os


class CorrectImage(object):

    def __init__(self, image_path):
        self.pic = image_path

    def _load_image(self):
        filename = os.path.join(self.pic, 'initial.png')
        camera = io.imread(filename)
