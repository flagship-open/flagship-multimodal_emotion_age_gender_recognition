import requests

#Session
sess = requests.Session()

#Setting
ip_address = 'http://0.0.0.0:9993'

input_speech = 'test_dis.wav'
output_path =  '.'

#Request
req = sess.post(ip_address, data={'input_speech': input_speech, 'output_path': output_path})

if req.status_code == 200:
    print(req,'success')
    print(req.content)

else:
    print('fail')



