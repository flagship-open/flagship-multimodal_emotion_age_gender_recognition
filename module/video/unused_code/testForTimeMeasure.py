import time
import numpy as np
from test import AdditionalInfo, calc_mode, test_input, test_face_cropper, test_age_model, test_gender_model,\
    test_cnn_model, test_lstm_model, test_post_processing, test_preprocessing

init_time = time.time()
end_time = time.time() - init_time

addition_info_instance = AdditionalInfo()


def main():
    # number of batch size(or fps)
    num_frame = 30
    k = test_input('./samples', num_frame)
    feature_list = []
    age_res_list = []
    gender_res_list = []

    frame_time_list = []
    frame_crop_time = 0
    frame_preprocessing_time = 0
    frame_gender_time = 0
    frame_age_time = 0
    frame_cnn_time = 0

    # for each intruder, face crop, preprocessing and feature extraction is done

    num_success_face_detect = 0

    for i in range(len(k)):
        curr_time = time.time()
        res_cropped, cnt = test_face_cropper(k[i])
        num_success_face_detect += cnt
        frame_crop_time = time.time() - curr_time

        curr_time = time.time()
        res_preprocessing = test_preprocessing(res_cropped)
        frame_preprocessing_time = time.time() - curr_time

        curr_time = time.time()
        res_cnn = test_cnn_model(res_preprocessing)
        frame_cnn_time = time.time() - curr_time

        curr_time = time.time()
        res_gender, res_gender_pred = test_gender_model(res_preprocessing)[0]
        frame_gender_time = time.time() - curr_time

        curr_time = time.time()
        res_age = test_age_model(res_preprocessing)
        frame_age_time = time.time() - curr_time

        age_res_list.append(res_age)
        gender_res_list.append((res_gender, res_gender_pred))

        if len(feature_list) == 0:
            feature_list = res_cnn[0]
        else:
            feature_list = np.concatenate((feature_list, res_cnn[0]))

        frame_time_list.append((frame_crop_time, frame_preprocessing_time,
                                frame_cnn_time, frame_gender_time, frame_age_time))

        curr_time = time.time()

    # for finishing whole video stream, then use LSTM to predict emotion from the sequential features
    # calculate final gender

    if num_success_face_detect < int(0.5 * len(k) * num_frame):
        print('Number of detected faces: ' + str(num_success_face_detect))
        print('Not enough face detected...')
        return None

    gender_final, gender_prob_final = calc_mode(gender_res_list)[0]
    gender_label = addition_info_instance.gender_label[gender_final]

    curr_time = time.time()

    # calculate final age
    age_final = test_post_processing(int(np.mean(age_res_list)))

    # calculate final emotion
    feature_list_npy = np.array(feature_list)
    res = test_lstm_model(feature_list_npy)

    deter_time = time.time() - curr_time

    for i in range(len(frame_time_list)):
        print('Frame ', i)
        print('Frame total time: ',
              frame_time_list[i][0] + frame_time_list[i][1] + frame_time_list[i][2] + frame_time_list[i][3] + frame_time_list[i][
                  4])
        print('Crop time:', frame_time_list[i][0])
        print('Preprocessing time:', frame_time_list[i][1])
        print('CNN time:', frame_time_list[i][2])
        print('Gender time:', frame_time_list[i][3])
        print('Age time:', frame_time_list[i][4])

    print("determine processing: ", deter_time)
    print("Model load time: ", end_time)

main()
