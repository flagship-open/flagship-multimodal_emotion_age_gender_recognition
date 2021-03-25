# Flagship 5차년도 멀티모달(Text,음성,영상)기반 감정, 나이, 성별 인식기 (~10/30)

### 작성자: 신영훈(KAIST)
모듈별 작성자: Text 김동성(한양대) / Audio 최신국(KAIST) / Video 이협우(KAIST) / Video RTMP 신영훈(KAIST) / Multimodal Multitask 신영훈(KAIST)

#### Note

* (2020/05/30) 5월 0.5 버전 업로드

* (2020/06/18) 6월 0.9 버전 업로드(텍스트 인식기 작동 안됨 -> 문제 해결)

* (2020/08/07) 8월 1.0 버전 업로드(영상 전처리 부분 속도 개선)

* (2020/08/18) 8월 2.0 버전 업로드(RTMP 부분 변경)

* (2020/09/31) 9월 3.0 버전 업로드(End-to-end base-model)

* (2020/10/30) 10월 4.0 버전 업로드(Attention based end-to-end multimodal)

* (2020/11/03) 10월 4.1 버전 업로드(Attention based end-to-end multimodal)

* (2021/03/25) module>speech>model, module>text>model, module>video>weights 학습 파일은 별도 요청 필요 (작성자: 김희성)
***

#### System/SW Overview

* 텍스트 인식기
  * input: input text (string)
  * output: recognition result (.json)
* 영상 전처리 인식기
  * Input: image folder (dir)
  * Output: Cropped face images (Tx224x224x3), error code (0~6)
* 영상 인식기 
  * Input: Face images (T[4fps]x224x224x3) , 
  * output: Cropped feature T[4fps]x4096 (.npy), recognition result(.json)
* 음성 인식기 
	* Input: wav file (.mp3/.wav)
	* Output: speech feature N(1/2fps)x768 (.npy), recognition result(.json)
* 멀티모달 인식기
  * Input: text, speech, video features, recognition result(.json)
  * Output: recognition result(.json)

***

#### How to Install

* pip install -r requirements.txt (requirements.txt file in each modal)

***

#### Main requirement (except single modal)

* python==3.5
* keras==2.2.4
* tensorflow==1.12.0
* reqests_toolbelt==0.9.1
* Flask==1.1.2
* facenet-pytorch==2.3.0

***

#### Network Architecture and features

* Model

  * 감정 인식기
    * 아래 세 입력 결과를 Fusion 하여 멀티모달 인식 결과 출력
      * 텍스트인식기의 bottleneck feature: 1x512 (npy)
      * 영상인식기의 CNN(VGG Face) feature Tx4096 (T는 전체 영상의 선택된 frame 수, 초당 4 frame씩 입력)
      * 음성인식기의 CNN feature Nx768 (N은 2 second 단위의 speech 입력)
  * 성별, 나이 인식기
    * 아래 두 입력 결과를 Fusion 하여 멀티모달 인식 결과 출력
      * 영상인식기의 FaceNet으로부터 출력된 age 값, gender 값  
      * 음성 인식기의 CNN 모델로부터 출력된 age 값, gender 값 

* Evaluation
  * 감정 인식
    * 7개의 label을 바탕으로 label별 prediction된 확률값을 출력한다.
    * 가장 높은 확률값을 가진 label을 Ground truth 표정의 label과 비교하여 correct / incorrect를 결정한다.
    * 모든 class에 대한 accuracy로 최종 evaluation을 수행한다.
  * 나이 인식
    * Predict된 나이값에 대하여 실제 나이와의 정해진 오차를 통해(ex) +-5살) 그 오차 안에 실제 나이가 들어올 경우 correct, 벗어날 경우 incorrect으로 평가한다.
  * 성별 인식
    * 2개의 label(male, female)을 바탕으로 label별 prediction된 확률값을 바탕으로 가장 큰 확률값을 label로 한다.
    * Ground truth 표정의 레이블과 비교하여 correct / incorrect를 결정한다.
    * 모든 class에 대한 accuracy로 최종 evaluation을 수행한다.

***
#### Quick start

* Step 0: 설치
  
  * 각 모듈별 라이브러리 설치: `pip3 install -r requirements.txt`
* Step 1: Module API 실행

  * Module API
    * 한번에 전체 실행: `python3.5 api_all.py`
    * 개별 실행
      * 텍스트인식: `python3.5 module/text/text_flask.py --gpu 1 --port 9991` 
	    * 음성인식: `python3.5 module/speech/speech_flask.py --gpu 1 --port 9992` 
	    * 영상 인식: `python3.5 module/video/video_flask.py --gpu 2 --port 9991` 
	    * 멀티모달 인식: `python3.5 module/mulitmodal/multi_flask.py --gpu 3 --port 9994` 
* Step2: Test API 실행
	  * RTMP 모듈 먼저 실행
	  * 인식기 테스트: `python3.5 recognition_client.py --input_text "나는 지금 너무 슬퍼" --input_video `
* Step3: 최종 결과 확인
  * 최종 결과는 json 형태의 (string, float)으로 출력이 되며 다음과 같이 출력된다.
    ```{ ('10001': 1), (10002: '0.1678'), (10003: '0.1315'), (10004: '0.1726'), (10005: '0.1722'), (10006: '0.1926'), (10007: '0.0694'), (20000: '25'), (30001: '0.4688'), (30002: '0.5311')}```



***

#### Training Data

   * Dataset used for multimodal training: KAIST 2018 / KAIST 2019 / KAIST 2020 / KAIST 2020_A / KAIST 2020_B / KAIST 2020_C
   * Labels in dataset: Emotion / Age / Gender

***

#### Validation metrics calculation
  * Dataset: Training: 80% / Test: 20% 
  * Accuaray: Correct number / Total Data number
***

#### HTTP-server API description (will be added until 10/30)


* **path, parameter, response를 명시한다.**

> *  /test_api/v1/actions/get_response
> * JSON parameters are:

> |Parameter|Type|Description|
> |---|---|---|
> |context|list of strings|List of previous messages from the dialogue history (max. 3 is used)|
> |emotion|string, one of enum|One of {'neutral', 'anger', 'joy', 'fear', 'sadness'}. An emotion to condition the response on. Optional param, if not specified, 'neutral' is used|

> * Request
> ```
> POST /test_api/v1/actions/get_response
> data: {
> 'context': ['Hello', 'Hi!', 'How are you?'],
> 'emotion': 'joy'
> }
> ```

> * Response OK
> ```
> 200 OK
> {
>  'response': 'I\'m fine!'
> }
> ```

***

#### Repository overview
> will be added until 10/30
***

#### configuration settings 
