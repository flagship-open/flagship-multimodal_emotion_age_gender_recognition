import keras
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
import os
import time
import numpy as np
import librosa
import json
import scipy
from pyvad import trim
from collections import OrderedDict


# For Check Time
bb = time.time()

# Load Model
module_path = os.path.dirname(os.path.realpath(__file__))
Model_Merge = load_model(module_path + '/model/Model_emotion_age_gender.h5')
Model_Merge._make_predict_function()
Model_Merge.summary()

# Load Feature Extraction Model
Model_Features = Model_Merge.get_layer('model_features')
Model_Features._make_predict_function()

# For Check Time2
start_time = time.time()

### Pre-processing
Num_Frame = 1500    # max wave length (15 sec)
Stride = 0.01       # stride (10ms)
Window_size = 0.025 # filter window size (25ms)
Num_data = 1
Num_mels = 40       # Mel filter number
pre_emphasis = 0.97  # Pre-Emphasis filter coefficient
Num_Crop_Frame = 200  # Frame size of crop

def preprocessing(y, sr):

    # Resampling to 16kHz
    if sr != 16000:
        sr_re = 16000  # sampling rate of resampling
        y = librosa.resample(y, sr, sr_re)
        sr = sr_re

    # Denoising
    y[np.argwhere(y == 0)] = 1e-10
    y_denoise = scipy.signal.wiener(y, mysize=None, noise=None)

    # Pre Emphasis filter
    y_Emphasis = np.append(y_denoise[0], y_denoise[1:] - pre_emphasis * y_denoise[:-1])

    # Normalization (Peak)
    y_max = max(y_Emphasis)
    y_Emphasis = y_Emphasis / y_max  # VAD 인식을 위해 normalize

    # Voice Activity Detection (VAD)
    vad_mode = 2  # VAD mode = 0 ~ 3
    y_vad = trim(y_Emphasis, sr, vad_mode=vad_mode, thr=0.01)  ## VAD 사용하여 trim 수행
    if y_vad is None:
        y_vad = y_Emphasis

    # De normalization
    y_vad = y_vad * y_max

    # Obtain the mel spectrogram
    S = librosa.feature.melspectrogram(y=y_vad, sr=sr, hop_length=int(sr * Stride), n_fft=int(sr * Window_size), n_mels=Num_mels, power=2.0)

    EPS = 1e-8
    S = np.log(S + EPS)
    r, Frame_length = S.shape

    # Obtain the normalized mel spectrogram
    S_norm = (S - np.mean(S)) / np.std(S)

    # zero padding
    Input_Mels = np.zeros((r, Num_Frame), dtype=float)
    if Frame_length < Num_Frame:
        Input_Mels[:, :Frame_length] = S_norm[:, :Frame_length]
    else:
        Input_Mels[:, :Num_Frame] = S_norm[:, :Num_Frame]

    # Input_Mels = np.expand_dims(Input_Mels, axis=0)
    # Input_Mels = np.transpose(Input_Mels, (0, 2, 1))
    # Input_Mels = np.expand_dims(Input_Mels, axis=-1)

    return Input_Mels, Frame_length


def Crop_Mels(Input_Mels_origin,Each_Frame_Num):
    Input_Mels_origin = Input_Mels_origin.T

    Crop_stride = int(Num_Crop_Frame / 2)

    # Calculate the number of cropped mel-spectrogram
    if Each_Frame_Num > Num_Frame:
        Number_of_Crop = int(Num_Frame / Crop_stride) - 1
    else:
        if Each_Frame_Num < Num_Crop_Frame:
            Number_of_Crop = 1
        else:
            Number_of_Crop = int(round(Each_Frame_Num / Crop_stride)) - 1

    ## Crop

    Cropped_Mels = np.zeros((Number_of_Crop, Num_Crop_Frame, Input_Mels_origin.shape[1]))
    crop_num = 0  # Crop된 data의 number
    if Each_Frame_Num > Num_Frame:  # If the frame number is higher than 1500, the number of crop is 14
        Each_Crop_Num = int(Num_Frame / Crop_stride) - 1
        for N_crop in range(0, Each_Crop_Num):
            Cropped_Mels[crop_num, :, :] = Input_Mels_origin[N_crop * Crop_stride:N_crop * Crop_stride + Num_Crop_Frame, :]
            crop_num += 1
    else:
        if Each_Frame_Num < Num_Crop_Frame:    # If the frame number is lower than 200, the number of crop is 1
            Cropped_Mels[crop_num, :, :] = Input_Mels_origin[:Num_Crop_Frame, :]
            crop_num += 1
        else:
            Each_Crop_Num = int(round(Each_Frame_Num / Crop_stride)) - 1    # Calculate the number of crop
            if round(Each_Frame_Num / Crop_stride) < Each_Frame_Num / Crop_stride:
                for N_crop in range(0, Each_Crop_Num):
                    Cropped_Mels[crop_num, :, :] = Input_Mels_origin[N_crop * Crop_stride:N_crop * Crop_stride + Num_Crop_Frame, :]
                    crop_num += 1
            else:
                for N_crop in range(0, Each_Crop_Num - 1):
                    Cropped_Mels[crop_num, :, :] = Input_Mels_origin[N_crop * Crop_stride:N_crop * Crop_stride + Num_Crop_Frame, :]
                    crop_num += 1
                shift_frame = int((Each_Frame_Num / Crop_stride - round(Each_Frame_Num / Crop_stride)) * Crop_stride)
                Cropped_Mels[crop_num, :, :] = Input_Mels_origin[(Each_Crop_Num - 1) * Crop_stride + shift_frame:(Each_Crop_Num - 1) * Crop_stride + shift_frame + Num_Crop_Frame,:]
                crop_num += 1

    return Cropped_Mels, Number_of_Crop

# Main Code
def generate(wav_name):

    # File Dir = Load from client.py(json)

    y, sr = librosa.load(wav_name)
    # Preprocessing(Resampling, Normalization, Denoising, Pre-emphasis, VAD)
    Input_Mels, Frame_length = preprocessing(y, sr)

    # Crop mel-spectrogram
    Cropped_Mels, Number_of_Crop = Crop_Mels(Input_Mels, Frame_length)
    Cropped_Mels = np.reshape(Cropped_Mels, (Cropped_Mels.shape[0], Cropped_Mels.shape[1], Cropped_Mels.shape[2], 1))

    # Speech Feature Extraction
    features = Model_Features.predict(Cropped_Mels)

    # Predict the cropped mels_log : Emotion, Age, Gender, Speaker
    [y_E_pred_crop, y_A_pred_crop, y_G_pred_crop] = Model_Merge.predict(Cropped_Mels)

    # Average the cropped prediction
    y_E_pred = np.mean(y_E_pred_crop, axis=0)     # Emotion
    y_A_pred = np.mean(y_A_pred_crop, axis=0)     # Age
    y_G_pred = np.mean(y_G_pred_crop, axis=0)     # Gender
    # y_S_pred = np.mean(y_S_pred_crop, axis=0)     # Speaker

    # Age Denormalize
    mean_age, std_age = 38.36, 11.01
    y_A_pred = y_A_pred * std_age + mean_age    # Age prediction value

    # age group
    y_A_pred_group = np.zeros((3,))

    if y_A_pred >= 40:  # 40s
        y_A_pred_group[2] = 1
    elif y_A_pred >= 30:  # 30s
        y_A_pred_group[1] = 1
    elif y_A_pred >= 20:  # 20s
        y_A_pred_group[0] = 1

    # Ready for data
    result = OrderedDict()

    # Emotion (Happiness:10001, Anger:10002, Disgust:10003, Fear:10004, Neutral:10005, Sadness:10006, Surprise:10007)
    result["10001"] = round(float(y_E_pred[0]),4)
    result["10002"] = round(float(y_E_pred[1]),4)
    result["10003"] = round(float(y_E_pred[2]),4)
    result["10004"] = round(float(y_E_pred[3]),4)
    result["10005"] = round(float(y_E_pred[4]),4)
    result["10006"] = round(float(y_E_pred[5]),4)
    result["10007"] = round(float(y_E_pred[6]),4)

    # Age (Predicted Age:20000, 20s:20003, 30s:20004, 40s:20005)
    result["20000"] = round(float(y_A_pred),4)
    result["20003"] = round(float(y_A_pred_group[0]),4)
    result["20004"] = round(float(y_A_pred_group[1]),4)
    result["20005"] = round(float(y_A_pred_group[2]),4)

    # Gender (Male:30001, Female:30002)
    result["30001"] = round(float(y_G_pred[0]),4)
    result["30002"] = round(1-float(y_G_pred[0]),4)

    # # Spekaer Embedding
    # Speaker_Emb = Model_Speaker_Emb.predict(Cropped_Mels)

    return result, features

