import os
import numpy as np
import h5py
from settings import *


def get_loader(dataset):
    def load_features(video_ids):
        features = list()
        shape = (-1, FRAMES_PER_VIDEO*2+1, 1024)
        for video_id in video_ids:
            features.append(dataset.get_multimodal_features(video_id))
        return np.array(features).reshape(shape)
    return load_features
