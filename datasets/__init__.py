import numpy as np
import pandas as pd

from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from settings import TEST_VAL_SIZE, BATCHSIZE

class Dataset(object):
    """Abstract dataset class

    video_aux_info
        -- load videos metadata (length and frame rate)
    get_audio_filename
        -- return audio file location by object/video id
    get_video_filename
        -- return video file location by object/video id
    get_spectrogram_filename
        -- return location of precomputed audio spectrogram by object/video id
    get_frames_filename
        -- return location of preprocessed frames by object/video id
    """
    meta_location = None
    audio_storage = None
    video_storage = None
    spectrogram_storage = None
    frame_storage = None
    n_classes = None

    def __init__(self):
        dataset = pd.read_csv(self.meta_location, names=["youtube_id", "class_id"])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            list(dataset.youtube_id),
            to_categorical(np.array(dataset.class_id, dtype=int)),
            test_size=TEST_VAL_SIZE)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            self.X_test,
            self.y_test,
            test_size=0.5)
        self.samples_per_epoch = len(self.X_train)
        self.nb_val_samples = len(self.X_val)
        self.data_loader = None

    def _get_filename(self, video_id, datatype, model=None):
        if datatype == 'audio':
            return self.get_audio_filename(video_id)
        elif datatype == 'video':
            return self.get_video_filename(video_id)
        elif datatype == 'spectrograms':
            return self.get_spectrogram_filename(video_id, model)
        elif datatype == 'frames':
            return self.get_frames_filename(video_id, model)
        else:
            return None

    def get_audio_filename(self, video_id):
        pass

    def get_video_filename(self, video_id):
        pass

    def get_spectrogram_filename(self, video_id):
        pass

    def get_frames_filename(self, video_id):
        pass

    def _batch_generator(self, inputs, targets, batch_size=BATCHSIZE, shuffle=False, endless=True):
        assert len(inputs) == len(targets)
        while True:
            if shuffle:
                indices = np.arange(len(inputs))
                np.random.shuffle(indices)
            for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
                if shuffle:
                    excerpt = indices[start_idx:start_idx + batch_size]
                else:
                    excerpt = slice(start_idx, start_idx + batch_size)
                yield np.array(self.data_loader(
                    np.array(inputs)[excerpt])), np.tile(np.array(
                    targets)[excerpt], 1).reshape(-1, self.n_classes)
            if not endless:
                break

    def batch_generator_train(self, endless=True):
        return self._batch_generator(inputs=self.X_train, targets=self.y_train,
                                     shuffle=True, endless=endless)

    def batch_generator_val(self):
        return self._batch_generator(inputs=self.X_val, targets=self.y_val,
                                     batch_size=1, shuffle=False, endless=False)


    def batch_generator_test(self):
        return self._batch_generator(inputs=self.X_test, targets=self.y_test,
                                     batch_size=1, shuffle=False, endless=False)

