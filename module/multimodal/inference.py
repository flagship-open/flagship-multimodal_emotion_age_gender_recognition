from keras.models import load_model
import numpy as np
from collections import OrderedDict
import json
import os
import glob

# load model

try:

    module_path = os.path.dirname(os.path.realpath(__file__))

    emotion_name = sorted(glob.glob(module_path + '/model/emotion_*'), reverse=True)
    emotion_model = load_model(emotion_name[0])
    emotion_model.summary()
    age_name = sorted(glob.glob(module_path + '/model/age_*'), reverse=True)
    age_model = load_model(age_name[0])
    age_model.summary()
    gender_name = sorted(glob.glob(module_path + '/model/gender*'), reverse=True)
    gender_model = load_model(gender_name[0])
    gender_model.summary()

except Exception as e:
    print("model load error:", e)


def predict(root_path):

    text_feature = np.load(root_path + 'text_features.npy')
    video_feature = np.load(root_path + 'video_features.npy')
    speech_feature = np.load(root_path + 'speech_features.npy')

    
    video_tmp = np.zeros((20, 4096))
    if(video_feature.shape[0]>20):
        video_tmp = video_feature[20,:]
    else:
        video_tmp[:video_feature.shape[0],:] = video_feature

    speech_tmp = np.zeros((5, 768))
    if(speech_feature.shape[0]>5):
        speech_tmp = speech_feature[:5,:]
    else:
        speech_tmp[:speech_feature.shape[0],:] = speech_feature
 
    text_feature = np.expand_dims(text_feature, 0)
    video_feature = np.expand_dims(video_tmp, 0)
    speech_feature = np.expand_dims(speech_tmp, 0)

   
    text_emotion = np.expand_dims(np.load(root_path + 'text_emotion.npy'),0)
    video_emotion = np.expand_dims(np.load(root_path + 'video_emotion.npy'),0)
    speech_emotion = np.expand_dims(np.load(root_path + 'speech_emotion.npy'),0)

    video_age = np.expand_dims(np.load(root_path + 'video_age.npy'),0)
    speech_age = np.expand_dims(np.load(root_path + 'speech_age.npy'),0)

    video_gender = np.expand_dims(np.load(root_path + 'video_gender.npy'),0)
    speech_gender = np.expand_dims(np.load(root_path + 'speech_gender.npy'),0)

    try:
        emotion_pred = emotion_model.predict([text_feature, video_feature, speech_feature, text_emotion, video_emotion, speech_emotion])
        age_pred = age_model.predict([video_feature, speech_feature, video_age, speech_age])
        gender_pred = gender_model.predict([video_feature, speech_feature, video_gender, speech_gender])
    except Exception as e:
        print("predict error:", e)
        emotion_pred = np.zeros((1,7))
        age_pred = np.zeros((1,1))
        gender_pred = np.zeros((1,2))

    result = OrderedDict()

    # happiness
    result["10001"] = round(float(emotion_pred[0][0]),4)
    # anger
    result["10002"] = round(float(emotion_pred[0][1]),4)
    # disgust
    result["10003"] = round(float(emotion_pred[0][2]),4)
    # fear
    result["10004"] = round(float(emotion_pred[0][3]),4)
    # neutral
    result["10005"] = round(float(emotion_pred[0][4]),4)
    # sadness
    result["10006"] = round(float(emotion_pred[0][5]),4)
    # surprise
    result["10007"] = round(float(emotion_pred[0][6]),4)

    # Age (Predicted Age:20000, 20s:20003, 30s:20004, 40s:20005)
    result["20000"] = round(float(age_pred[0]),4)

    # Gender (Male:30001, Female:30002)
    result["30001"] = round(float(gender_pred[0][0]),4)
    result["30002"] = round(float(gender_pred[0][1]),4)

    return result


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Input data root path
    root_path = '../../samples/single_outputs/'
    print(predict(root_path))
