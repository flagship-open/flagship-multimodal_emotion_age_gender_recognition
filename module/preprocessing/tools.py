import os
import numpy as np
import cv2
from PIL import Image
import sys
from collections import Counter
from keras.applications.inception_v3 import preprocess_input
from facenet_pytorch import MTCNN
# https://www.kaggle.com/timesler/fast-mtcnn-detector-55-fps-at-full-resolution

DEBUG = False

class Preprocessor():
    def __init__(self):
        pass
    def process(self, inputList, resizedFactor=(224,224)):
        resList = []
        for eachInput in inputList:
            eachInputResized = cv2.resize(eachInput, resizedFactor)
            eachInputPreprocessed = preprocess_input(eachInputResized)
            resList.append(eachInputPreprocessed)
        return resList


class FaceCropperAll():
    def __init__(self, marginRatio=1.3, type='mtcnn', resizeFactor=0.5):
        '''
        initializer of face cropper all
        :param marginRatio: ratio of margin
        :param type: type of face cropper:('mtcnn', 'haar' )
        '''
        if(type == 'mtcnn'):
            self.detector = MTCNNFaceDetector()
        else:
            assert False, 'Wrong face cropper type...'

    def detectMulti(self, inputList):
        return self.detector.detectMulti(inputList)


class FaceCropper():
    def __init__(self, marginRatio=1.3, resizeFactor=1.0):
        '''
        FaceCropper Basic Class
        :param marginRatio: margin of face(default: 1.3)
        '''
        self.marginRatio=1.3
        self.prevX = 0
        self.prevY = 0
        self.prevW = 0
        self.prevH = 0
        self.resizeFactor = resizeFactor

    def cropFace(self, input, x,y,w,h):
        '''
        Crop Face with given bbox
        :param input: input image
        :param x: X
        :param y: Y
        :param w: W
        :param h: H
        :return: cropped image
        '''

        x_n = int(x - (self.marginRatio - 1) / 2.0 * w)
        y_n = int(y - (self.marginRatio - 1) / 2.0 * h)
        w_n = int(w * self.marginRatio)
        h_n = int(h * self.marginRatio)

        return input[y_n:y_n + h_n, x_n:x_n + w_n],x,y,w,h

    def detect(self, input):
        '''
        Face detect with single input
        :param input: single image (W x H x C)
        :return: bbox information(x,y,w,h)
        '''
        pass
    def detectMulti(self, inputList):
        '''
        Face detect with multiple inputs
        :param inputList: multi images (N x W x H x C)
        :return: face cropped image list (N x W' x H' x C)
        '''
        pass


class MTCNNFaceDetector(FaceCropper):
    def __init__(self, marginRatio=1.3, resizeFactor=1.0):
        super().__init__(marginRatio, resizeFactor)
        # load mtcnn model
        # self.mtcnnModel = MTCNN()
        self.mtcnn = MTCNN(select_largest=True, device='cuda:0')


    def detect(self, img):
        '''
        Face detect with single input
        :param img: single image (W x H x C)
        :return: face croppped image (W' x H' x C)
                cnt: number of successfully detected faces
        '''

        imgResize = cv2.resize(img, dsize=(0,0), fx=self.resizeFactor, fy=self.resizeFactor)

        res = self.mtcnnModel.detect_faces(imgResize)

        # if no faces detect, than return previous bbox position
        if(len(res) == 0):
            return self.prevX, self.prevY, self.prevW, self.prevH, False

        # process of margin ratio
        x,y,w,h = res[0]['box']

        # save the previous result
        self.prevX = x
        self.prevY = y
        self.prevW = w
        self.prevH = h

        return x,y,w,h, True

    def detectMulti(self, frame_list):
        '''
        Face detect with multiple inputs
        :param inputList: multi images (N x W x H x C)
        :return: face cropped image list (N x W' x H' x C), num face detected, num blur detected
        '''
        
        blur_thd = 30 # defualt: 40
        no_blur_rate = 0.4
        prob_thd = 0.999 # default = 0.999
        
        """
        error_code
        0: no error
        1: unkwon error
        2: blur occured (shaking)
        3: out of bound (or too close)
        4: distubed by somthing
        5: not enough face
        6: no face
        """
        
        error_code = 0
        if(DEBUG==True):
            print(len(frame_list))
        img_list = []
        no_blur_detected = 0
        
        for eachInput in frame_list:
            imgResize = cv2.resize(eachInput, dsize=(0,0), fx=self.resizeFactor, fy=self.resizeFactor)
            img8U = np.uint8(imgResize)

            # blur detection
            gray = cv2.cvtColor(img8U, cv2.COLOR_BGR2GRAY)
            blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()           
            if(blur_val < blur_thd):
                if(DEBUG==True):
                    print("blurry image")
            else:
                # face detection
                img = cv2.cvtColor(img8U, cv2.COLOR_BGR2RGB)
                img_list.append(img)
                no_blur_detected += 1

        face_detected = 0
        faces = []
        boundary_error_count = 0
        prop_error_count = 0

        if(no_blur_detected < no_blur_rate * len(frame_list)):
            error_code = 2
            if(DEBUG==True):
                print('not blurred image:', no_blur_detected)
            return faces, face_detected, error_code
        boxes, probs = self.mtcnn.detect(img_list)
        
        for i, frame in enumerate(img_list):
            box_ind = int(i)
            if boxes[box_ind] is None:
                continue
            for box in boxes[box_ind]:
                box = [int(b) for b in box]
                
                # Face detection error case
                if(frame.shape[0] < box[3] or frame.shape[1] < box[2] or box[1] < 0 or box[0] < 0):
                    boundary_error_count += 1
                    if(DEBUG==True):
                        print("face boundary error")
                    continue
                if (probs[box_ind][0] < prob_thd):
                    prop_error_count += 1
                    if(DEBUG==True):
                        print("face prob error:", probs[box_ind])
                    continue
                if(DEBUG==True):
                    cv2.imwrite('samples/img_detected'+str(i)+'.jpg',frame[box[1]:box[3], box[0]:box[2]])                
                faces.append(frame[box[1]:box[3], box[0]:box[2]])
                face_detected += 1
                
        if(boundary_error_count > int(0.5 * len(frame_list))):
            error_code = 3
        elif(prop_error_count > int(0.5 * len(frame_list))):
            error_code = 4
        else:
            error_code = 0
        
        return faces, face_detected, error_code


class additionalInfo():
    def __init__(self):
        self.emotionLabel = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
        self.genderLabel = ['Female', 'Male']

additionInfoInstance = additionalInfo()
detectorType = 'mtcnn'
detector = FaceCropperAll(type=detectorType, resizeFactor=1.0)
preprocessor = Preprocessor()

def input_frame(parentPath):
    '''
    test for sample input(M x N x W x H x C)
    M: number of intruders
    N: samples per one intruder(one second) (default: 10 fps)
    W: width of image
    H: height of image
    C: number of channel in image(3)
    :return: M x N x W x H x C inputs
    '''
    frameSeq = os.listdir(parentPath)
    frameSeq.sort()
    frame_list = []
    
    for i in range(len(frameSeq)):

        frameSeqAbsPath = os.path.join(parentPath, frameSeq[i])
        img = cv2.imread(frameSeqAbsPath)
        frame_list.append(img)

    return np.array(frame_list)


def face_cropper(inputList):
    global detector
    if (DEBUG == True):
        print('TEST FACE CROPPER')
    faces, face_detected, error_code = detector.detectMulti(inputList)
    if(faces is None):
        print('TEST FAIL FACE CROPPER')
        return None
    return faces, face_detected, error_code

def preprocessing(inputList):
    global preprocessor
    if (DEBUG == True):
        print('TEST PREPROCESSOR')
    res = preprocessor.process(inputList)
    if(res is None):
        print('TEST FAIL PREPROCESSOR')
        return None
    return res
