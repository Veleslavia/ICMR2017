import cv2
import os
import h5py
import numpy as np
import skimage.transform

from settings import *

input_tensor_shape = (-1, 224, 224, 3)

def get_loader(dataset):
    def load_video(video_ids):
        index = None
        frame_list = list()
        for video_id in video_ids:
            if len(video_id) == 2:
                video_id, index = video_id[0], video_id[1]
            frames_filename = dataset.get_frames_filename(video_id)
            if os.path.exists(frames_filename) and index:
                h5file = h5py.File(frames_filename, 'r')
                frame = np.array(h5file[index])
                if np.random.random() < 0.5:
                    frame = flip_axis(frame, 1)
                    frame = deep_pool_for_inception(frame)
                h5file.close()
                frame_list.append(np.array(frame))
            else:
                video_filename = dataset.get_video_filename(video_id)
                if video_id != 'demo':
                    video_length, video_fps = dataset.video_aux_info[video_id]
                else:
                    video_length, video_fps = 30, 24
                image_preprocessing = deep_pool_for_inception
                frames = get_frames(video_filename, video_length, video_fps, image_preprocessing)
                h5file = h5py.File(frames_filename, 'w')
                for i in range(len(frames)):
                    ds = h5file.create_dataset(str(i), (224, 224, 3), 'i', compression='lzf')
                    ds[...] = frames[i]
                h5file.close()
                frame_list = frames
        return np.array(frame_list).reshape(input_tensor_shape)
    return load_video


def get_frames(filename, length, video_fps, image_preprocessing_func):
    cap = cv2.VideoCapture(filename)
    frames = list()
    while True:
        cap.set(CAP_PROP_POS_MSEC, 0)
        frame_idx = 0
        ret = True
        while ret:
            ret, frame = cap.read()
            if ret:
                frame_idx += 1
                # if current frame number is divisible by (length * video_fps) / FRAMES_PER_VIDEO
                # and we take uniformly distributed frames
                if not (frame_idx % int((length * video_fps)/FRAMES_PER_VIDEO)):
                    frames.append(image_preprocessing_func(frame))
            if len(frames) == FRAMES_PER_VIDEO:
                cap.release()
                return np.array(frames).reshape(input_tensor_shape)


def deep_pool_for_inception(frame):
    if frame.shape != (224, 224):
        frame = skimage.transform.resize(frame, (256, 256), preserve_range=True)
        frame = frame[16:240, 16:240, :]
    frame /= 255.
    frame -= 0.5
    frame *= 2.
    return frame


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x
