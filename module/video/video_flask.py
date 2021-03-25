import os
import json
import time
import argparse
import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np

## Argparse
parser = argparse.ArgumentParser(description='Video server')
parser.add_argument('--port', default='9993', type=str, help='port number')
parser.add_argument('--gpu', default='0', type=str, help='gpu number')

args = parser.parse_args()
port = args.port
gpu = args.gpu


# For GPU setting
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from video_inference import predict

# For Flask API

init_time = time.time()
app = Flask(__name__)
graph = tf.get_default_graph()

@app.route('/', methods=['POST', 'GET'])

def video_recognition():
    global graph 
    init_time = time.time()

    with graph.as_default():

        input_video = request.form['input_video']
        output_path = request.form['output_path']
        print(input_video, output_path)

        result = 0
        emotion = np.zeros((7))
        age = np.zeros((1))
        gender = np.zeros((2))
        features = np.zeros((1,4096))
 
        try:
            result, features = predict(input_video)
          
            emotion[0] = result["10001"]
            emotion[1] = result["10002"]
            emotion[2] = result["10003"]
            emotion[3] = result["10004"]
            emotion[4] = result["10005"]
            emotion[5] = result["10006"]
            emotion[6] = result["10007"]

            age[0] = result["20000"]

            gender[0] = result["30001"]
            gender[1] = result["30002"]
            print(result)

        except Exception as e:
            print("video_error:", e)

    np.save(output_path + 'video_features.npy', features)
    np.save(output_path + 'video_emotion.npy', emotion)
    np.save(output_path + 'video_age.npy', age)
    np.save(output_path + 'video_gender.npy', gender)

    return jsonify(result)

if __name__ == '__main__':
    print('pre-loading takes {}s'.format(time.time() - init_time))
    app.run(host='0.0.0.0', port=port, threaded=True, debug=False)
