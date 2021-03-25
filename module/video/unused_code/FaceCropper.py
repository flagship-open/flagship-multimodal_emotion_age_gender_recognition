from mtcnn.mtcnn import MTCNN
from cv2 import resize, cvtColor, COLOR_BGR2GRAY, CascadeClassifier, CASCADE_SCALE_IMAGE
from numpy import array, uint8

'''
Face Cropper total
'''


class FaceCropperAll:
    def __init__(self, cropper_type='haar'):
        """
        initializer of face cropper all
        """

        if cropper_type == 'mtcnn':
            self._detector = MTCNNFaceDetector()
        elif cropper_type == 'haar':
            self._detector = HaarCascadeFaceDetector()
        else:
            assert False, 'Wrong face cropper type...'

    def detect_multi(self, input_list):
        return self._detector.detect_multi(input_list)


"""
Basic face cropper
"""


class FaceCropper:
    def __init__(self, margin_ratio=1.3, resize_factor=1.0):
        """
        FaceCropper Basic Class
        :param margin_ratio: margin of face(default: 1.3)
        :param resize_factor: resize factor(default: 1.0)
        """

        self._margin_ratio = margin_ratio
        self._resize_factor = resize_factor
        self._prevX = 0
        self._prevY = 0
        self._prevW = 0
        self._prevH = 0


    def crop_face(self, img, x, y, w, h):
        """
        Crop Face with given bbox
        :param img: input image
        :param x: X
        :param y: Y
        :param w: W
        :param h: H
        :return: cropped image
        """

        x_n = int(x - (self._margin_ratio - 1) / 2.0 * w)
        y_n = int(y - (self._margin_ratio - 1) / 2.0 * h)
        w_n = int(w * self._margin_ratio)
        h_n = int(h * self._margin_ratio)

        return img[y_n:y_n + h_n, x_n:x_n + w_n], x, y, w, h

    def detect(self, img):
        """
        Face detect with single input
        :param img: single image (W x H x C)
        :return: bbox information(x,y,w,h)
        """
        pass

    def detect_multi(self, input_list):
        """
        Face detect with multiple inputs
        :param input_list: multi images (N x W x H x C)
        :return: face cropped image list (N x W' x H' x C)
        """
        pass


"""
MTCNN Face Detector
"""


class MTCNNFaceDetector(FaceCropper):
    def __init__(self, margin_ratio=1.3, resize_factor=1.0):
        super().__init__(margin_ratio, resize_factor)
        # load mtcnn model
        self._mtcnnModel = MTCNN()

    def detect(self, img):
        """
        Face detect with single input
        :param img: single image (W x H x C)
        :return: face croppped image (W' x H' x C)
                cnt: number of successfully detected faces
        """

        img_resize = resize(img, dsize=(0, 0),
                            fx=self._resize_factor, fy=self._resize_factor)

        res = self._mtcnnModel.detect_faces(img_resize)

        # if no faces detect, than return previous bbox position
        if len(res) == 0:
            return self._prevX, self._prevY, self._prevW, self._prevH, False

        # process of margin ratio
        x, y, w, h = res[0]['box']

        # save the previous result
        self._prevX = x
        self._prevY = y
        self._prevW = w
        self._prevH = h

        return x, y, w, h, True

    def detect_multi(self, input_list):
        """
        Face detect with multiple inputs
        :param input_list: multi images (N x W x H x C)
        :return: face cropped image list (N x W' x H' x C)
        """
        res_list = []
        cnt = 0
        for eachInput in input_list:
            bbox = self.detect(eachInput)
            if bbox is None:
                return None
            x, y, w, h, is_detected = bbox

            if is_detected:
                cnt += 1

            res = self.crop_face(eachInput, x, y, w, h)

            res_list.append(res[0])

        return array(res_list), cnt


"""
OPENCV Haar Cascade Face Detector
"""


class HaarCascadeFaceDetector(FaceCropper):
    def __init__(self, margin_ratio=1.3, resize_factor=1.0):
        super().__init__(margin_ratio, resize_factor)

        # load xml file
        self._faceCascade = CascadeClassifier('weights/haarcascade_frontalface_default.xml')
        # self.faceCascade = cv2.CascadeClassifier('weights/haarcascade_frontalface_alt_tree.xml')

    def detect(self, img):
        """
        Face detect with single input
        :param img: single image (W x H x C)
        :return: face cropped image(W' x H' x C)
        """
        img_resize = resize(img, dsize=(0, 0),
                            fx=self._resize_factor, fy=self._resize_factor)
        img_8u = uint8(img_resize)

        gray = cvtColor(img_8u, COLOR_BGR2GRAY)

        res = self._faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=CASCADE_SCALE_IMAGE)

        if len(res) == 0:
            return self._prevX, self._prevY, self._prevW, self._prevH, False

        max_value = -999999

        max_x = 0
        max_y = 0
        max_w = 0
        max_h = 0

        for f in res:
            (x, y, w, h) = [v for v in f]
            size = w * h
            if size > max_value:
                max_value = size
                (max_x, max_y, max_w, max_h) = (x, y, w, h)

        self._prevX = max_x
        self._prevY = max_y
        self._prevW = max_w
        self._prevH = max_h

        return max_x, max_y, max_w, max_h, True

    def detect_multi(self, input_list):
        """
        Face detect with multiple inputs
        :param input_list: multi images(N x W x H x C)
        :return: face cropped image list(N x W' x H' x C)
                cnt: number of successing face detect
        """

        res_list = []
        cnt = 0
        for eachInput in input_list:
            bbox = self.detect(eachInput)

            x, y, w, h, is_detected = bbox
            if is_detected:
                cnt += 1
            res = self.crop_face(eachInput, x, y, w, h)
            res_list.append(res[0])

        return array(res_list), cnt
