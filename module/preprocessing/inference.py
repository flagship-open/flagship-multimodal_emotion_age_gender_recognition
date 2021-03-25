from tools import face_cropper
from tools import preprocessing
from tools import input_frame
import numpy as np

def predict(input_dir):
    """
    # input: input images dir (10 fps)
    # return: pre_processed_images, error_code
    ## error_code
        0: no error
        1: unkwon error
        2: blur occured (shaking)
        3: out of bound (or too close)
        4: distubed by somthing
        5: not enough face
        6: no face
    """

    frame_list = input_frame(input_dir)
    cropped_frame, num_face_detected, error_code = face_cropper(frame_list)
    res_frame = []

    # selected frames: total frame * 0.4
    num_select_frames = int(0.4 * len(frame_list))
        
    # Error case
    if(error_code != 0):
        return res_frame, error_code
    else:
        if(num_face_detected == 0):
            error_code = 6
            return res_frame, error_code
        if(num_face_detected < num_select_frames):
            error_code = 5
            return res_frame, error_code

    # Select 4 frames
    selected_idx = []
    ran_num = np.random.randint(0, num_face_detected)

    for i in range(num_select_frames):
        while ran_num in selected_idx:
            ran_num = np.random.randint(0,num_face_detected)
        selected_idx.append(ran_num)
    selected_idx.sort()

    selected_frame = []
    for i in range(len(selected_idx)):
        selected_frame.append(cropped_frame[i])

    res_frame = preprocessing(selected_frame)

    return res_frame, error_code


