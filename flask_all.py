import subprocess
import psutil
import time


lists = []


try:
    lists.append(subprocess.Popen(["python3", 'module/text/text_flask.py', '--port', '9991', '--gpu', '0']))
    lists.append(subprocess.Popen(["python3", 'module/video/video_flask.py', '--port', '9992', '--gpu', '1']))
    lists.append(subprocess.Popen(["python3", 'module/speech/speech_flask.py', '--port', '9993', '--gpu', '0']))
    lists.append(subprocess.Popen(["python3", 'module/multimodal/multi_flask.py', '--port', '9994', '--gpu', '2']))

    while 1:
        time.sleep(0.1)

except KeyboardInterrupt:
    print("terminated")
    lists[0].terminate()
    for i in range(len(lists)):
        lists[i].terminate()
