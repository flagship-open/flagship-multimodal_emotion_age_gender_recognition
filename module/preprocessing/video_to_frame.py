import cv2
import csv
import os
import glob
import threading


def video_to_frame(load_dir, save_dir):
 
    fps = 3
    num_frames_per_folder = 10

    video_file_path = sorted(glob.glob(load_dir))

    for i in range(len(video_file_path)):

        print(i)

        cap = cv2.VideoCapture(video_file_path[i])

        name = video_file_path[i].split('/')[-1].split('.')[0]
        save_folder_path = save_dir + '/' + str(name)

        print(i, name)

        count = 0
        
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:

                if(count%(fps*num_frames_per_folder)==0):
                    folder_num=int(count/(fps*num_frames_per_folder))
                    folder_name = save_folder_path  + '/' + str(folder_num).zfill(5)
                    if not (os.path.isdir(folder_name)):
                        os.makedirs(os.path.join(folder_name))

                if(count%fps==0):
                    save_file_path =  folder_name + '/' + str(count).zfill(5) + '.jpg'
                    if (frame.shape[0]<frame.shape[1]):
                        frame=cv2.resize(frame,dsize=(720,1280))
                    cv2.imwrite(os.path.join(save_file_path), frame)

                count += 1

            else:
                break


if __name__ == "__main__":

    load_dir = '/data4/yhshin/dataset/flagship/KAIST_Multimodal_2019/*.mp4'
    save_dir = '/data4/yhshin/dataset/flagship/frames/2019'
    video_to_frame(load_dir, save_dir)
    print("Done")

    load_dir = '/data4/yhshin/dataset/flagship/KAIST_Multimodal_2020_A/*.mp4'
    save_dir = '/data4/yhshin/dataset/flagship/frames/2020_A'
    video_to_frame(load_dir, save_dir)
    print("Done")

    load_dir = '/data4/yhshin/dataset/flagship/KAIST_Multimodal_2020_B/*.mp4'
    save_dir = '/data4/yhshin/dataset/flagship/frames/2020_B'
    video_to_frame(load_dir, save_dir)
    print("Done")

    load_dir = '/data4/yhshin/dataset/flagship/KAIST_Multimodal_2020_C/*.mp4'
    save_dir = '/data4/yhshin/dataset/flagship/frames/2020_C'
    video_to_frame(load_dir, save_dir)
    print("Done")


