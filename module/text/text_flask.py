import os
import tensorflow as tf
import json
import time
import argparse
import numpy as np
from flask import Flask, jsonify, request, Response, make_response


parser = argparse.ArgumentParser(description='Text server')
parser.add_argument('--port', default='9991', type=str, help='port number')
parser.add_argument('--gpu', default='0', type=str, help='gpu number')

args = parser.parse_args()

gpu = args.gpu
port = args.port

os.environ["CUDA_VISIBLE_DEVICES"]= gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from text_inference import predict


init_time = time.time()
app = Flask(__name__)
graph = tf.get_default_graph()

@app.route('/', methods=['POST', 'GET'])

def text_recognition():
    global graph
    with graph.as_default():

        init_time = time.time()
        print('request.form : {}'.format(request.form).encode('utf8'))
        input_text = request.form['input_text']
        output_path = request.form['output_path']
        print(input_text)

        result = 0
        emotion = np.zeros((7))
        try:
            result, features = predict(input_text)
    
            emotion[0] = result["10001"]
            emotion[1] = result["10002"]
            emotion[2] = result["10003"]
            emotion[3] = result["10004"]
            emotion[4] = result["10005"]
            emotion[5] = result["10006"]
            emotion[6] = result["10007"]
        except:
            print("text error")

    np.save(output_path + 'text_features.npy', features)
    np.save(output_path + 'text_emotion.npy', emotion)

    print(result)
    print('it takes {:.2f}s'.format(time.time() - init_time))

    return jsonify(result)

if __name__ == '__main__':
    print('pre-loading takes {}s'.format(time.time() - init_time))
    app.run(host='0.0.0.0', port=9991) ## Change port number
