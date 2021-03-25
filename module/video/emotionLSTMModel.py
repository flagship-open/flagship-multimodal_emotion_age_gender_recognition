"""
LSTM model
frames emotion predictor
"""

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Dropout


class EmotionLSTMModel:
    def __init__(self, model_file, feature_size=4096):
        self.__emotionLabel = ['Anger', 'Disgust', 'Fear', 'Happiness',
                               'Neutral', 'Sadness', 'Surprise']
        metrics = ['accuracy']
        self.__feature_size = feature_size
        self.__num_class = 7
        self.__model = self.model_definition()

        self.__model.load_weights(model_file)

        optimizer = Adam(lr=0.01)

        self.__model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)

        print(self.__model.summary())

    def model_definition(self):
        """
        Model Definition(No json file available, so all architectures should be revealed
        :return: model
        """
        model = Sequential()
        model.add(LSTM(self.__feature_size, return_sequences=False,
                       input_shape=(None, self.__feature_size), dropout=0.5))

        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.__num_class, activation='softmax'))

        return model

    def predict(self, input_list):
        """
        Predict emotion from the inputList
        :param input_list: M x featureSize(4096)
        :return: M x 7
        """
        res = self.__model.predict(input_list)
        return res
