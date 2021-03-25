# Flagship 5���⵵ �����ν� ��ó�� RTMP aAPI

### ����: �ſ���

#### Note

* (2020/08/19) ver 1.0 ���ε�
* (2020/10/07) ver 2.0 ���ε�
* (2020/10/29) ver 3.0 ���ε�

***

#### System/SW Overview

* ���� ��ó�� �νı�: rtmp image folder

***

#### How to Install

***

#### Main requirement

* python \>= 3.5
* keras==2.2.4
* tensorflow == 1.12.0
* reqests_toolbelt

***

#### Quick start(RTMP ����)

* Step 0: ��ġ
  * Same as Video module library

* Step 1: flask API ����
  *����:  `python3.5 pp_flask.py`
	
* Step 2: API �׽�Ʈ
  *����: `pp_client.py' 
	* ����: input_video_dir, output_dir , num_second�� api�� ����
	    * input_video_dir: �Է� �����ӵ��� ����� ����
	    * output_dir: ��ó���� ��°��� ����� ����
	    * num_second: ���� �ð� (��)
  *���
    * ��ó���� ���� output_dir������
	* error_code �� return


#### Error code �� front-end ���� �ȳ���
    0: no error (�ַ� �߻� ����) 
    1: unknown error (�����Ұ����� �ַ�): ��ȭ�� �ٽ� �õ��� �ּ���.
    2: blur occured (shaking) (������ ��鸮�ų� ������ �ȸ´� ���): ȭ���� �ʹ� ���� ��鸮�ų� ������ ���մϴ�. �ٽ� �õ����ּ���.
    3: out of bound (or too close) (�ʹ� �����ų� ���� ���� ������ ��� ���): ���� �ʹ� �����ų� ȭ�鿡 ������ �ʽ��ϴ�. �ٽ� �õ��� �ּ���.
    4: hidden by somthing (���� �ٸ� ��ü�� ���� ������ ���): ��ȭ�� ���� ���������ϴ�. �ٽ� �õ��� �ּ���.
    5: not enough face detected (��ü frame���� ���� �ʹ� ���� ����� ���): ���� ������� �ʾҽ��ϴ�. �ٽ� �õ��� �ּ���.
    6: no face (���� �ƿ� ������� ���� ���): ���� ������� �ʾҽ��ϴ�. �ٽ� �õ��� �ּ���.


#### Repository overview
> * `flaskRTMP.py` - Flask ���� ��� �ڵ�
> * `infer.py` - ���� ��� �ڵ�
> * `rtmp_api_test.py` - Sample data�� �׽�Ʈ�� ���� �ڵ� 

