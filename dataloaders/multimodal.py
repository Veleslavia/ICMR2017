import numpy as np
import h5py
from settings import *


def get_loader(dataset):
    def load_features(video_ids):
        features = list()
        for video_id in video_ids:
            video_features = list()
            feature_filename = dataset.get_features_filename(video_id)
            try:
                data_file = h5py.File(feature_filename, 'r')
            except:
                print video_id
            for key in data_file.keys():
                frame_features = np.array(data_file[key])
                video_features.append(frame_features)
            data_file.close()
            features.append(video_features)
        return features
    return load_features
