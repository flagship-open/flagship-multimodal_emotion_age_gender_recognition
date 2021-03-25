import requests


if __name__ =='__main__':

    input_text = "나는 행복해요"
    feature_path = "."

    try:
        sess = requests.Session()
        text_address = 'http://0.0.0.0:9997'
        req = sess.post(text_address, data={'input_text': input_text, 'output_path': feature_path})

        if req.status_code == 200:
            print(req, 'text_success')
        else:
            print('fail')
    except:
            print('Text module connection error')
