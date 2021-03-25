# Test API

import numpy as np
from test import test_input, test_face_cropper, test_cnn_model, test_lstm_model,\
    test_age_model, test_gender_model, test_post_processing, test_preprocessing
from test import addition_info_instance, calc_mode


def build_json(gender_res, age_res, emo_res):
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
    video_emo = {}
    for i in range(len(emo_code_list)):
        video_emo[emo_code_list[i]] = str(emo_res[emo_order_list[i]])
    video_gen = {}
    for i in range(len(gender_code_list)):
        video_gen[gender_code_list[i]] = str(gender_prob_list[i])
    video_age = {}
    for i in range(len(age_code_list)):
        video_age[age_code_list[i]] = str(age_value_list[i])

    res_dict = {'Video_Emo': video_emo, 'Video_Age': video_age, 'Video_Gen': video_gen}

    print(res_dict)

    return res_dict


def test_api_inner(seq_path):
    """
    Test API inner
    :param seq_path: parent path of the image sequences
    :return: json string
    """
    # fps
    num_frame = 30
    k = test_input(seq_path, num_frame)
    feature_list = []
    age_res_list = []
    gender_res_list = []

    num_success_face_detect = 0

    # for each intruder, face crop, preprocessing and feature extraction is done
    for i in range(len(k)):
        # face cropper
        res_cropped, cnt = test_face_cropper(k[i])
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
    emo_final = test_lstm_model(feature_list_npy)

    # build json file
    json_str = build_json((gender_label, gender_prob_final), age_final, emo_final)

    return json_str


def test_api():
    with open('testInput') as readFile:
        all_line = readFile.readlines()
    sequence_path = all_line[0].strip()

    test_api_inner(sequence_path)


if __name__ == '__main__':
    test_api()
