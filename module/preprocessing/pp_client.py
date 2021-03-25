import requests
import time

root_path = ''

""" Num Second """
num_second = "001"

""" Data path """
input_video_dir = root_path + 'samples/img_good_case/'
output_dir = root_path + 'samples/'  # single modal feature and result path

init_time = time.time()

try:
    sess = requests.Session()
    address = 'http://0.0.0.0:9990' # Change address
    req = sess.post(address, data={'input_path': input_video_dir, 'num_second': num_second,'output_path': output_dir})
    if req.status_code ==200:
        print(req,'rtmp processing success')
        error_code=(req.content)
        print('error:',error_code)
    else:
        print('fail')
except:
    print('Preprocessing connection error')

end_time = time.time() - init_time
print('RTMP pre-processing time:', round(end_time, 4))


