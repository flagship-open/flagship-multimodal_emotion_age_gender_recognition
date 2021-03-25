# Flagship 5차년도 동영상 기반 나이,성별,표정 인식기

### 작성자: 이협우

#### Note

* (2020/04/27) 4월 마스터 버전 업로드
* (2020/06/03) 5월 TestAPI를 위한 script 추가, Sample sequences 파일 추가
* (2020/07/27) flask server 테스트 스크립트 추가 및 flask server의 주소 및 구조 변경
* (2020/10/16) 최종 코드 업데이트

***

#### System/SW Overview

* 나이, 성별 인식기: FaceNet + Triplet Loss
* 표정 인식기: CNN+LSTM

***

#### How to Install

* pip install -r requirements.txt

***

#### Main requirement

* python \>= 3.5
* Keras==2.2.4
* Keras-Aplication==1.0.8
* Keras-Preprocessing==1.1.0
* mtcnn==0.0.9
* numpy==1.16.4
* opencv-contrib-python==4.1.2.30
* opencv-python==4.1.0.25
* scikit-learn==0.21.3
* scipy=1.3.0
* tensorflow-estimator==1.14.0
* tensorflow-gpu=1.12.0
* Flask=1.1.2

***

#### Network Architecture and features

* Model
  * 나이 인식기
    * FaceNet에서 Scale-varying Triplet Ranking with Classification Loss를 이용, Adience, IMDB 등의 public dataset으로부터 학습
  * 성별 인식기
    * FaceNet을 이용, Transfer learning으로 Adience으로부터 학습
  * 표정 인식기
    * LSTM + CNN 기반의 동영상(이미지 시퀀스) 기반 표정 인식 모듈
    * CNN: AffectNet으로부터 fine tuning을 한 vggFace 네트워크로써, 뒷단의 4096길이의 fc layer까지 feed forward 수행
    * LSTM: Flagship 3rd 데이터로부터 학습을 한 LSTM 네트워크; CNN 네트워크로부터 4096의 feature vector를 입력으로 하여 인식 수행
* Evaluation
  * 나이 인식
    * Predict된 나이값에 대하여 실제 나이와의 정해진 오차를 통해(ex) +-5살) 그 오차 안에 실제 나이가 들어올 경우 correct, 벗어날 경우 incorrect으로 평가한다.
  * 성별 인식
    * 2개의 label(male, female)을 바탕으로 label별 prediction된 확률값을 바탕으로 가장 큰 확률값을 표정 label로 한다.
    * Ground truth 표정의 레이블과 비교하여 correct / incorrect를 결정한다.
    * 모든 class에 대한 accuracy로 최종 evaluation을 수행한다.
  * 표정 인식
    * 7개의 label을 바탕으로 label별 prediction된 확률값을 바탕으로 가장 큰 확률값을 표정 label로 한다.
    * Ground truth 표정의 레이블과 비교하여 correct / incorrect를 결정한다.
    * 모든 class에 대한 accuracy로 최종 evaluation을 수행한다.

***

#### Quick start

* Step 0: 설치

  ```
  pip install -r requirements.txt
  ```

* Step 1: TestAPI 파일

  * Test할 sequence의 선택

    * testInput 파일에서 테스트해볼 sequence의 parent folder path를 입력한다.(ex)

    * ```
      samples/seqA
      ```

  * TestAPI.py 실행

    * ```
      python TestAPI.py
      ```

  * 최종 결과

    * 최종 결과는 json 형태의 string으로 출력이 되며 다음과 같이 출력된다.

    * ```
      {'Video_Emo': {10001: '0.093600184', 10002: '0.16787936', 10003: '0.13155556', 10004: '0.17263636', 10005: '0.17228816', 10006: '0.19260682', 10007: '0.06943359'}, 'Video_Gen': {30001: '0.4688760042190552', 30002: '0.531124'}, 'Video_Age': {20000: '25'}}
      ```

      형태의 string으로 출력이 되며 다음과 같이 출력된다.
  
#### Flask 서버 테스트
* Step 1: 서버 실행

  ```
  python flaskMain.py
  ```

* Step 2-1: TestFlask.py 실행

  ```
  python TestFlask.py
  ```
* Step 2-2: 웹브라우저에서 실행  
  
    
  다음의 url로 접속하여 결과 확인

  http://127.0.0.1:6009/seqPath/test

* Step 3: 임의의 sequence에 대한 테스트

  임의의 sequence 테스트시, samples 폴더 및에 sequence 이미지를 폴더 형태로 저장하고 다음으로 접속하면 결과를 확인할 수 있다.
  
  http://127.0.0.1:6009/seqPath/폴더명
  
  

#### Training Data
***

#### Training Model

***

#### Validation metrics calculation

***

#### HTTP-server API description

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

> * `samples/` – 테스트용 이미지 sequences 위치; 현재 seqA와 seqB가 존재
> * `weights` – 학습한 네트워크의 weight를 저장(나이, 성별, 표정 모두 존재)
> * `age` – Age prediction 관련 테스트 코드
> * `gender` – Gender prediction 관련 테스트 코드
> * `TestAPI.py` - API Test를 위한 스크립트
> * `testForTimeMeasure.py` - API에서 Time 측정을 위한 스크립트
> * `flaskMain.py` - Flask 메인 통신 코드

***

#### configuration settings 
