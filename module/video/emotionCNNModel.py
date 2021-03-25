"""
CNN model
static image feature extractor
"""
from keras import models
from numpy import expand_dims, zeros, shape, array
from tensorflow import get_default_graph


class EmotionCNNModel:
    def __init__(self, model_file, batch_size=32,
                 input_shape=(224, 224), pop_layer_num=6):
        self.__input_shape = input_shape
        self.__batch_size = batch_size

        # load model
        json_name = model_file + '.json'

        with open(json_name) as json_file:
            loaded_model_json = json_file.read()

        self.__model = models.model_from_json(loaded_model_json)
        self.__model.load_weights(model_file, True)

        # eliminate layer to use fc layer 6
        for k in range(pop_layer_num):
            self.__model.layers.pop()

        self.__model.outputs = [self.__model.layers[-1].output]
        self.__model.output_layers = [self.__model.layers[-1]]
        self.__model.layers[-1].outbound_nodes = []

        self.__graph = get_default_graph()

    def extract_single(self, img):
        """
        extract feature of an img
        :param img: input image(224,224,3)
        :return: fc layer (4096)
        """
        x = expand_dims(img, axis=0)
        features = self.__model.predict([x, zeros(shape(img)[0])])
        return features[0]

    def extract_multi(self, img_list):
        """
        extract feature of imgList by batch
        first build batch then test
        :param img_list: list of input image(M, 224, 224, 3)
        :return: list of fc layer(M, 4096)
        """
        img_count = len(img_list)
        batch_list = []
        res_list = []
        with self.__graph.as_default():
            for idx in range(img_count):
                batch_list.append(img_list[idx])

                if (idx+1) % self.__batch_size == 0:
                    x = array(batch_list)
                    res = self.__model.predict([x, zeros(shape(x)[0])])
                    res_list.append(res)
                    batch_list = []
            if img_count < self.__batch_size:
                x = array(batch_list)
                res = self.__model.predict([x, zeros(shape(x)[0])])
                res_list.append(res)

        return array(res_list)
