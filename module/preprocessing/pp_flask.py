import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]= "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess= tf.Session(config=config)

from inference import predict
import numpy as np
import time
from flask import Flask, request, make_response

app = Flask(__name__)
graph = tf.get_default_graph()
init_time = time.time()

@app.route('/', methods=['POST', 'GET'])
def video_preprocessing():
    global graph
    with graph.as_default():
        input_video = request.form['input_path']
        num_second = request.form['num_second'] 
        output_path = request.form['output_path']

        try:
            features, error_code = predict(input_video)
            if(error_code == 0):
                np.save(output_path + 'video_preprop_' + num_second + '.npy',features)
                print("accept")
            else:
                print("reject")
        except:
            print("unknown error")
            error_code = 1
            
    return str(error_code)

# Run
if __name__ == '__main__':
    input_dir = "samples/img_good_case"
    print('pre-loading time:', time.time()-init_time)
    print(predict(input_dir))
    app.run(host='0.0.0.0', port=9990, threaded=True, debug=False)
