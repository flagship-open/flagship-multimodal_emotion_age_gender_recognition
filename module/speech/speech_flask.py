import tensorflow as tf
import os
import argparse
import time
import numpy as np
import json
from flask import Flask, request, jsonify

parser = argparse.ArgumentParser(description='Speech server')
parser.add_argument('--port', default='9993', type=str, help='port number')
parser.add_argument('--gpu', default='0', type=str, help='port number')

args = parser.parse_args()

gpu = args.gpu
port = args.port

os.environ["CUDA_VISIBLE_DEVICES"] = gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from speech_inference import generate

# For Check Time
init_time = time.time()

# For Flask API
app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])

def speech_recognitionn():
    print('request.form : {}'.format(request.form).encode('utf8'))
    input_speech = request.form['input_speech']
    output_path = request.form['output_path']
    aa = time.time()
    result, features = generate(input_speech)

    features = features.reshape(-1, 768)

    emotion = np.zeros((7))
    emotion[0] = result["10001"]
    emotion[1] = result["10002"]
    emotion[2] = result["10003"]
    emotion[3] = result["10004"]
    emotion[4] = result["10005"]
    emotion[5] = result["10006"]
    emotion[6] = result["10007"]

    age = np.zeros((1))
    age[0] = result["20000"]

    gender = np.zeros((2))
    gender[0] = result["30001"]
    gender[1] = result["30002"]

    np.save(output_path + 'speech_features.npy', features)
    np.save(output_path + 'speech_emotion.npy', emotion)
    np.save(output_path + 'speech_age.npy', age)
    np.save(output_path + 'speech_gender.npy', gender)

    print(result)
    print('it takes {:.2f}s'.format(time.time() - aa))
    return jsonify(result)

if __name__ == '__main__':
    print('pre-loading takes {}s'.format(time.time() - init_time))
    app.run(host='0.0.0.0', port=port)
