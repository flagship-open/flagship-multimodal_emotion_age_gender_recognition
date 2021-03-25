import os
import numpy as np
import cv2

#from FaceCropper import MTCNNFaceDetector
#from FaceCropper import FaceCropperAll
#from Preprocessing import Preprocessor
from emotionCNNModel import EmotionCNNModel
from emotionLSTMModel import EmotionLSTMModel
from collections import Counter
import recognize_gender
import recognize_age_morph

DEBUG = False


class AdditionalInfo:
    def __init__(self):
        self.emotion_label = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
        self.gender_label = ['Female', 'Male']


addition_info_instance = AdditionalInfo()

age_model = recognize_age_morph.Runner()
gender_model = recognize_gender.Runner()

# detector type; haar or mtcnn
#detector_type = 'haar'
#detectorType = 'mtcnn'

#detector = FaceCropperAll(cropper_type=detector_type)
#preprocessor = Preprocessor()
module_path = os.path.dirname(os.path.realpath(__file__))
cnn_model = EmotionCNNModel(module_path + '/weights/AffectNet_scratch_best.h5')
lstm_model = EmotionLSTMModel(module_path + '/weights/lstm-features.sample.hdf5')
"""
if DEBUG:
    print('Fake Batch...')

zero_list = []

for i in range(32):
    zero_list.append(np.zeros((224, 224, 3)))
if DEBUG:
    print('Fake face crop')
if detector_type != 'haar':
    detector.detect_multi(zero_list)
if DEBUG:
    print('Fake cnn Model')
cnn_model.extract_multi(zero_list)
if DEBUG:
    print('Fake gender Model')
gender_model.recognize_gender(zero_list[0])
if DEBUG:
    print('Fake age Model')
age_model.recognize_age(zero_list[1])
if DEBUG:
    print('Fake batch done...')
"""


def calc_mode(res_list):
    """
    Calculate the mode
    :param res_list: list of values for calculating mode
    :return: result 1
    """

    counts = Counter(res_list)

    max_count = max(counts.values())
    prob_max_count = max_count / len(counts)
    max_count_label = [x_i for x_i, count in counts.items() if count == max_count]

    return max_count_label[0], prob_max_count


def test_post_processing(age_val):
    """
    Post Processing the output of predicted age value
    :return: 25 if age_val <= 25, 45 if age_val >= 46
    """
    if age_val <= 25:
        return 25
    elif age_val >= 46:
        return 46


def test_input(parent_path, num_frame=30):
    """
    test for sample input(M x N x W x H x C)

    M: number of intruders
    N: samples per one intruder(one second)(default: 30(30 fps))
    W: width of image
    H: height of image
    C: number of channel in image(3)

    :return: M x N x W x H x C inputs
    """

    frame_seq = os.listdir(parent_path)
    frame_seq.sort()

    whole_input_list = []
    intruder_input_list = []
    for idx in range(len(frame_seq)):

        frame_seq_abs_path = os.path.join(parent_path, frame_seq[idx])

        img = cv2.imread(frame_seq_abs_path)

        intruder_input_list.append(img)

        if (i + 1) % num_frame == 0:
            whole_input_list.append(intruder_input_list)
            intruder_input_list = []

    whole_input_list.append(intruder_input_list)

    return np.array(whole_input_list)


def test_face_cropper(input_list):
    global detector
    if DEBUG:
        print('TEST FACE CROPPER')
    res, cnt = detector.detect_multi(input_list)
    if res is None:
        print('TEST FAIL FACE CROPPER')
        return None
    return res, cnt


def test_preprocessing(input_list):
    global preprocessor
    if DEBUG:
        print('TEST PREPROCESSOR')
    res = preprocessor.process(input_list)
    if res is None:
        print('TEST FAIL PREPROCESSOR')
        return None
    return res


def test_cnn_model(input_list):
    global cnn_model
    if DEBUG:
        print("TEST CNN Model")
    res = cnn_model.extract_multi(input_list)
    if res is None:
        print('TEST FAIL CNN Model')
        return None
    return res


def test_gender_model(input_list):
    global gender_model
    res_list = []
    if DEBUG:
        print('TEST GENDER Model')
    for each_input in input_list:
        pred_gen, prob_gen = gender_model.recognize_gender(each_input)
        res_list.append((pred_gen, prob_gen))

    return calc_mode(res_list)


def test_age_model(input_list):
    global age_model
    res_list = []
    if DEBUG:
        print('TEST AGE Model')
    for eachInput in input_list:
        pred_age = age_model.recognize_age(eachInput)
        res_list.append(pred_age)
    res_list = np.array(res_list)
    rep_age = int(np.mean(res_list))
    return rep_age


def test_lstm_model(input_list):
    global lstm_model
    if DEBUG:
        print('TEST LSTM Model')
    input_list_expand = np.expand_dims(input_list, axis=0)
    res = lstm_model.predict(input_list_expand)
    if res is None:
        print('TEST FAIL LSTM Model')
        return None
    return res[0]


def main(seq_path):
    # fps
    num_frame = 30
    k = test_input(seq_path, num_frame)
    feature_list = []
    age_res_list = []
    gender_res_list = []

    num_success_face_detect = 0

    # for each intruder, face crop, preprocessing and feature extraction is done
    for idx in range(len(k)):
        # face cropper
        res_cropped, cnt = test_face_cropper(k[idx])
        num_success_face_detect += cnt

        # preprocessing
        res_preprocessing = test_preprocessing(res_cropped)

        # feature extractor
        res_cnn = test_cnn_model(res_preprocessing)

        # gender processing
        res_gender, res_gender_pred = test_gender_model(res_preprocessing)[0]

        # age processing
        res_age = test_age_model(res_preprocessing)

        age_res_list.append(res_age)
        gender_res_list.append((res_gender, res_gender_pred))

        if len(feature_list) == 0:
            feature_list = res_cnn[0]
        else:
            feature_list = np.concatenate((feature_list, res_cnn[0]))

    if num_success_face_detect < int(0.5 * len(k) * num_frame):
        print('Number of detected faces: ' + str(num_success_face_detect))
        print('Not enough face detected...')
        return None

    # for finishing whole video stream, then use LSTM to predict emotion from the sequential features
    # calculate final gender

    gender_final, gender_prob_final = calc_mode(gender_res_list)[0]
    gender_label = addition_info_instance.gender_label[gender_final]

    # calculate final age
    # post processing for age value
    age_final = test_post_processing(int(np.mean(age_res_list)))

    # calculate final emotion
    feature_list_npy = np.array(feature_list)
    res = test_lstm_model(feature_list_npy)

    print(gender_label, age_final, res)


