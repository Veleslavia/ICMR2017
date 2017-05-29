import os
import numpy as np
import h5py
from settings import *


def get_loader(dataset):
    def load_features(video_ids):
        features = list()
        shape = (-1, FRAMES_PER_VIDEO*2+1, 1024)
        for video_id in video_ids:
            feature_filename = dataset.get_features_filename(video_id)
            if os.path.exists(feature_filename):
                data_file = h5py.File(feature_filename, 'r')
                video_features = data_file[video_id+'/video'].value
                audio_features = data_file[video_id+'/audio'].value
                data_file.close()
                video_features = video_features.reshape((-1, 1024))
                features.append(np.concatenate((video_features, audio_features)))
            else:
                print "Prepare features as for the multimodal experiment." \
                      "You need to save audio and video features from penultimate " \
                      "layers of corresponding audio and video models." \
                      "Check an example in ./dataloaders/multimodal.py "
                """
                from keras import backend as K
                import h5py

                features_filename = h5py.File('./demo/demo.multimodal.h5', 'w')

                model = audio_model
                dataset = prefered_dataset.setup_data_loader(model.__name__)

                activation_func = K.function([model.input] + [K.learning_phase()], [model.layers[-2].output])
                audio_features = activation_func([dataset.data_loader(['demo']), 1.])[0]
                ds = features_filename.create_dataset('demo'+'/'+'audio', (1, 1024), 'f', compression='lzf')
                ds[...] = audio_features

                model = video_model
                dataset = prefered_dataset.setup_data_loader(model.__name__)
                activation_func = K.function([self.model.input] + [K.learning_phase()], [model.layers[-2].output])
                video_features = activation_func([dataset.data_loader(['demo']), 1.])[0]
                ds = features_filename.create_dataset('demo'+'/'+'video', (FRAMES_PER_VIDEO, 2048), 'f',
                compression='lzf')
                ds[...] = video_features

                features_filename.close()

                """
        return np.array(features).reshape(shape)
    return load_features
