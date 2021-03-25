# Flagship 5Â÷³âµµ ¸ÖÆ¼¸ð´Þ(Text,À½¼º,¿µ»ó)±â¹Ý °¨Á¤, ³ªÀÌ, ¼ºº° ÀÎ½Ä±â (~10/30)

### ÀÛ¼ºÀÚ: ½Å¿µÈÆ(KAIST)
¸ðµâº° ÀÛ¼ºÀÚ: Text ±èµ¿¼º(ÇÑ¾ç´ë) / Audio ÃÖ½Å±¹(KAIST) / Video ÀÌÇù¿ì(KAIST) / Video RTMP ½Å¿µÈÆ(KAIST) / Multimodal Multitask ½Å¿µÈÆ(KAIST)

#### Note

* (2020/05/30) 5¿ù 0.5 ¹öÀü ¾÷·Îµå

* (2020/06/18) 6¿ù 0.9 ¹öÀü ¾÷·Îµå(ÅØ½ºÆ® ÀÎ½Ä±â ÀÛµ¿ ¾ÈµÊ -> ¹®Á¦ ÇØ°á)

* (2020/08/07) 8¿ù 1.0 ¹öÀü ¾÷·Îµå(¿µ»ó ÀüÃ³¸® ºÎºÐ ¼Óµµ °³¼±)

* (2020/08/18) 8¿ù 2.0 ¹öÀü ¾÷·Îµå(RTMP ºÎºÐ º¯°æ)

* (2020/09/31) 9¿ù 3.0 ¹öÀü ¾÷·Îµå(End-to-end base-model)

* (2020/10/30) 10¿ù 4.0 ¹öÀü ¾÷·Îµå(Attention based end-to-end multimodal)

* (2020/11/03) 10¿ù 4.1 ¹öÀü ¾÷·Îµå(Attention based end-to-end multimodal)

* (2021/03/25) module>speech>model, module>text>model, module>video>weights ¿ ¿¿¿ ¿¿¿ ¿¿ ¿¿(¿¿¿: ¿¿¿)
  
***

#### System/SW Overview

* ÅØ½ºÆ® ÀÎ½Ä±â
  * input: input text (string)
  * output: recognition result (.json)
* ¿µ»ó ÀüÃ³¸® ÀÎ½Ä±â
  * Input: image folder (dir)
  * Output: Cropped face images (Tx224x224x3), error code (0~6)
* ¿µ»ó ÀÎ½Ä±â 
  * Input: Face images (T[4fps]x224x224x3) , 
  * output: Cropped feature T[4fps]x4096 (.npy), recognition result(.json)
* À½¼º ÀÎ½Ä±â 
	* Input: wav file (.mp3/.wav)
	* Output: speech feature N(1/2fps)x768 (.npy), recognition result(.json)
* ¸ÖÆ¼¸ð´Þ ÀÎ½Ä±â
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

  * °¨Á¤ ÀÎ½Ä±â
    * ¾Æ·¡ ¼¼ ÀÔ·Â °á°ú¸¦ Fusion ÇÏ¿© ¸ÖÆ¼¸ð´Þ ÀÎ½Ä °á°ú Ãâ·Â
      * ÅØ½ºÆ®ÀÎ½Ä±âÀÇ bottleneck feature: 1x512 (npy)
      * ¿µ»óÀÎ½Ä±âÀÇ CNN(VGG Face) feature Tx4096 (T´Â ÀüÃ¼ ¿µ»óÀÇ ¼±ÅÃµÈ frame ¼ö, ÃÊ´ç 4 frame¾¿ ÀÔ·Â)
      * À½¼ºÀÎ½Ä±âÀÇ CNN feature Nx768 (NÀº 2 second ´ÜÀ§ÀÇ speech ÀÔ·Â)
  * ¼ºº°, ³ªÀÌ ÀÎ½Ä±â
    * ¾Æ·¡ µÎ ÀÔ·Â °á°ú¸¦ Fusion ÇÏ¿© ¸ÖÆ¼¸ð´Þ ÀÎ½Ä °á°ú Ãâ·Â
      * ¿µ»óÀÎ½Ä±âÀÇ FaceNetÀ¸·ÎºÎÅÍ Ãâ·ÂµÈ age °ª, gender °ª  
      * À½¼º ÀÎ½Ä±âÀÇ CNN ¸ðµ¨·ÎºÎÅÍ Ãâ·ÂµÈ age °ª, gender °ª 

* Evaluation
  * °¨Á¤ ÀÎ½Ä
    * 7°³ÀÇ labelÀ» ¹ÙÅÁÀ¸·Î labelº° predictionµÈ È®·ü°ªÀ» Ãâ·ÂÇÑ´Ù.
    * °¡Àå ³ôÀº È®·ü°ªÀ» °¡Áø labelÀ» Ground truth Ç¥Á¤ÀÇ label°ú ºñ±³ÇÏ¿© correct / incorrect¸¦ °áÁ¤ÇÑ´Ù.
    * ¸ðµç class¿¡ ´ëÇÑ accuracy·Î ÃÖÁ¾ evaluationÀ» ¼öÇàÇÑ´Ù.
  * ³ªÀÌ ÀÎ½Ä
    * PredictµÈ ³ªÀÌ°ª¿¡ ´ëÇÏ¿© ½ÇÁ¦ ³ªÀÌ¿ÍÀÇ Á¤ÇØÁø ¿ÀÂ÷¸¦ ÅëÇØ(ex) +-5»ì) ±× ¿ÀÂ÷ ¾È¿¡ ½ÇÁ¦ ³ªÀÌ°¡ µé¾î¿Ã °æ¿ì correct, ¹þ¾î³¯ °æ¿ì incorrectÀ¸·Î Æò°¡ÇÑ´Ù.
  * ¼ºº° ÀÎ½Ä
    * 2°³ÀÇ label(male, female)À» ¹ÙÅÁÀ¸·Î labelº° predictionµÈ È®·ü°ªÀ» ¹ÙÅÁÀ¸·Î °¡Àå Å« È®·ü°ªÀ» label·Î ÇÑ´Ù.
    * Ground truth Ç¥Á¤ÀÇ ·¹ÀÌºí°ú ºñ±³ÇÏ¿© correct / incorrect¸¦ °áÁ¤ÇÑ´Ù.
    * ¸ðµç class¿¡ ´ëÇÑ accuracy·Î ÃÖÁ¾ evaluationÀ» ¼öÇàÇÑ´Ù.

***
#### Quick start

* Step 0: ¼³Ä¡
  
  * °¢ ¸ðµâº° ¶óÀÌºê·¯¸® ¼³Ä¡: `pip3 install -r requirements.txt`
* Step 1: Module API ½ÇÇà

  * Module API
    * ÇÑ¹ø¿¡ ÀüÃ¼ ½ÇÇà: `python3.5 api_all.py`
    * °³º° ½ÇÇà
      * ÅØ½ºÆ®ÀÎ½Ä: `python3.5 module/text/text_flask.py --gpu 1 --port 9991` 
	    * À½¼ºÀÎ½Ä: `python3.5 module/speech/speech_flask.py --gpu 1 --port 9992` 
	    * ¿µ»ó ÀÎ½Ä: `python3.5 module/video/video_flask.py --gpu 2 --port 9991` 
	    * ¸ÖÆ¼¸ð´Þ ÀÎ½Ä: `python3.5 module/mulitmodal/multi_flask.py --gpu 3 --port 9994` 
* Step2: Test API ½ÇÇà
	  * RTMP ¸ðµâ ¸ÕÀú ½ÇÇà
	  * ÀÎ½Ä±â Å×½ºÆ®: `python3.5 recognition_client.py --input_text "³ª´Â Áö±Ý ³Ê¹« ½½ÆÛ" --input_video `
* Step3: ÃÖÁ¾ °á°ú È®ÀÎ
  * ÃÖÁ¾ °á°ú´Â json ÇüÅÂÀÇ (string, float)À¸·Î Ãâ·ÂÀÌ µÇ¸ç ´ÙÀ½°ú °°ÀÌ Ãâ·ÂµÈ´Ù.
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


* **path, parameter, response¸¦ ¸í½ÃÇÑ´Ù.**

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
