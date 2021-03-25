# Flagship 5차년도 영상인식 전처리 RTMP aAPI

### 수정: 신영훈

#### Note

* (2020/08/19) ver 1.0 업로드
* (2020/10/07) ver 2.0 업로드
* (2020/10/29) ver 3.0 업로드

***

#### System/SW Overview

* 영상 전처리 인식기: rtmp image folder

***

#### How to Install

***

#### Main requirement

* python \>= 3.5
* keras==2.2.4
* tensorflow == 1.12.0
* reqests_toolbelt

***

#### Quick start(RTMP 버전)

* Step 0: 설치
  * Same as Video module library

* Step 1: flask API 실행
  *실행:  `python3.5 pp_flask.py`
	
* Step 2: API 테스트
  *실행: `pp_client.py' 
	* 역할: input_video_dir, output_dir , num_second를 api로 전송
	    * input_video_dir: 입력 프레임들이 저장된 폴더
	    * output_dir: 전처리된 출력값이 저장된 폴더
	    * num_second: 영상 시간 (초)
  *출력
    * 전처리된 영상 output_dir에저장
	* error_code 값 return


#### Error code 별 front-end 송출 안내문
    0: no error (애러 발생 안함) 
    1: unknown error (예측불가능한 애러): 대화를 다시 시도해 주세요.
    2: blur occured (shaking) (영상이 흔들리거나 초점이 안맞는 경우): 화면이 너무 많이 흔들리거나 역광이 심합니다. 다시 시도해주세요.
    3: out of bound (or too close) (너무 가깝거나 얼굴이 범위 밖으로 벗어난 경우): 얼굴이 너무 가깝거나 화면에 들어오지 않습니다. 다시 시도해 주세요.
    4: hidden by somthing (얼굴이 다른 물체에 의해 가려진 경우): 대화중 얼굴이 가려졌습니다. 다시 시도해 주세요.
    5: not enough face detected (전체 frame에서 얼굴이 너무 조금 검출된 경우): 얼굴이 검출되지 않았습니다. 다시 시도해 주세요.
    6: no face (얼굴이 아예 검출되지 않은 경우): 얼굴이 검출되지 않았습니다. 다시 시도해 주세요.


#### Repository overview
> * `flaskRTMP.py` - Flask 메인 통신 코드
> * `infer.py` - 관련 모듈 코드
> * `rtmp_api_test.py` - Sample data의 테스트를 위한 코드 

