import requests
import time
from requests_toolbelt import threaded
import argparse

## argparse
parser = argparse.ArgumentParser(description='Multimodal API client')
parser.add_argument('--input_text', default='나는 지금 너무 슬퍼', type=str, metavar='PATH',
                                    help='input text data')
parser.add_argument('--input_video', default='../../samples/input/face_cropped_sample/', type=str, metavar='PATH',
                                    help='input video data')
parser.add_argument('--input_speech', default='../../samples/input/speech_sample.wav', type=str, metavar='PATH',
                                    help='input speech data')

parser.add_argument('--single_output', default='../../samples/single_outputs/', type=str, metavar='PATH',
                                    help='output data path')
parser.add_argument('--multi_output', default='../../samples/multi_outputs/', type=str, metavar='PATH',
                                    help='output data path')

parser.add_argument('--text_port', default='9991', type=str, help='port number')
parser.add_argument('--video_port', default='9992', type=str, help='port number')
parser.add_argument('--speech_port', default='9993', type=str, help='port number')
parser.add_argument('--multi_port', default='9994', type=str, help='port number')

args = parser.parse_args()

input_text = args.input_text
input_video = args.input_video
input_speech = args.input_speech
single_output_path = args.single_output
multi_output_path = args.multi_output

text_port = args.text_port
video_port = args.video_port
speech_port = args.speech_port
multi_port = args.multi_port

## Single modal Threading 
initTime = time.time()

urls = [{'url':'http://0.0.0.0:' + text_port, 'method':'POST', 'data':{'input_text': input_text, 'output_path': single_output_path}},
        {'url':'http://0.0.0.0:' + video_port, 'method':'POST', 'data':{'input_video': input_video, 'output_path': single_output_path}},
        {'url':'http://0.0.0.0:' + speech_port, 'method':'POST', 'data':{'input_speech': input_speech, 'output_path': single_output_path}}]

responses, resp_error = threaded.map(urls,num_processes=3)

print(responses)
print(resp_error)
endTime = time.time() - initTime
print('Single modal recognition time:', round(endTime,4))

## Multimodal module client
init_time = time.time()

try:
    sess = requests.Session()
    speech_address = 'http://0.0.0.0:' + multi_port
    req = sess.post(speech_address, data={'input_path': single_output_path, 'output_path': multi_output_path})
    if req.status_code == 200:
        print('mulitmodal result:', req.content)
    else:
        print('multimodal connection error')
except Exception as e:
    print('Multimodal error:', e)

print('Multimodal processing time:', round(time.time() - init_time, 4))
