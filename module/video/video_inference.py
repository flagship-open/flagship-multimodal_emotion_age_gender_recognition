import os
import numpy as np

import glob
from functions_video import test_input, test_face_cropper, test_cnn_model, test_lstm_model,\
    test_age_model, test_gender_model, test_post_processing, test_preprocessing
from functions_video import addition_info_instance, calc_mode
from collections import OrderedDict

def build_dict(gender_res, age_res, emo_res):
    """
    build json file from the network output(gender, age, emotion)
    :param gender_res: gender result( ex) ('Male', 0.8))
    :param age_res: age result( ex) 23 )
    :param emo_res: emotion result( ex) [0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.3])
    :return: json string
    """

    emo_code_list = [10001 + i for i in range(7)]
    emo_order_list = [3, 0, 1, 2, 4, 5, 6]
    gender_code_list = [30001 + i for i in range(2)]
    gender_prob_list = [1 - gender_res[1], gender_res[1]]
    age_code_list = [20000]
    age_value_list = [age_res]

    # build dictionary
    results = OrderedDict()
    for i in range(len(emo_code_list)):
        results[str(emo_code_list[i])] = round(float(emo_res[emo_order_list[i]]),4)
    for i in range(len(age_code_list)):
        results[str(age_code_list[i])] = int(age_value_list[i])
    for i in range(len(gender_code_list)):
        results[str(gender_code_list[i])] = round(float(gender_prob_list[i]),4)

    return results


def predict(seq_path):
    """
    Test API inner
    :param seq_path: parent path of the image sequences
    :return: json string
    """
    feature_name = sorted(glob.glob(seq_path +'*.npy'))
    """
    if not (feature_name):
        results = OrderedDict()
        results["10001"] = 0.14
        results["10002"] = 0.14
        results["10003"] = 0.14
        results["10004"] = 0.14
        results["10005"] = 0.16
        results["10006"] = 0.14
        results["10007"] = 0.14
        results["20000"] = 30
        results["30001"] = 0.5
        results["30002"] = 0.5
        feature_list_npy = np.zeros((1,4096))
        return results, feature_list_npy
    """

    age_res_list = []
    gender_res_list = []
    feature_list = []

    for i in range(len(feature_name)):

        # feature extractor
        res_preprocessing = np.load(feature_name[i])
        """
        if(res_preprocessing.shape[0]==0):
            results = OrderedDict()
            results["10001"] = 0.14
            results["10002"] = 0.14
            results["10003"] = 0.14
            results["10004"] = 0.14
            results["10005"] = 0.16
            results["10006"] = 0.14
            results["10007"] = 0.14
            results["20000"] = 30
            results["30001"] = 0.5
            results["30002"] = 0.5
            feature_list_npy = np.zeros((1,1,4096))
            return results, feature_list_npy
        """
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

    # for finishing whole video stream, then use LSTM to predict emotion from the sequential features
    # calculate final gender
    gender_final, gender_prob_final = calc_mode(gender_res_list)[0]
    gender_label = addition_info_instance.gender_label[gender_final]

    # calculate final age
    # post processing for age value
    age_final = test_post_processing(int(np.mean(age_res_list)))

    # calculate final emotion
    feature_list_npy = np.array(feature_list)
    emo_final = test_lstm_model(feature_list_npy)

    # build json file
    result = build_dict((gender_label, gender_prob_final), age_final, emo_final)

    return result, feature_list_npy


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    load_dir = "samples/"
    print(predict(load_dir))
