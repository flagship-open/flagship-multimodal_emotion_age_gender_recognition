"""
Preprocessor for input sequences
"""

from keras.applications.inception_v3 import preprocess_input
from cv2 import resize


class Preprocessor:
    def __init__(self, resize_factor=(224, 224)):
        self.__resize_factor = resize_factor
        pass

    def process(self, input_list):
        res_list = []

        for each_input in input_list:
            each_input_resized = resize(each_input, self.__resize_factor)
            each_input_preprocessed = preprocess_input(each_input_resized)
            res_list.append(each_input_preprocessed)
        return res_list
