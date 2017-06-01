import numpy as np
import pickle
from . import Dataset
from settings import FCVID_AUDIO_STORAGE, FCVID_VIDEO_STORAGE
from settings import FCVID_MULTIMODAL_FEATURES, FCVID_INFO_FILE, FCVID_META_FILE
from settings import DEMO_STORAGE, DEMO_AUDIO, DEMO_VIDEO, FRAMES_PER_VIDEO
from dataloaders import audio, video, multimodal


def setup_data_loader(model_module_name):
    dataset = FCVID()
    if ('crnn_classification' in model_module_name) or ('xception' in model_module_name):
        dataset.data_loader = audio.get_loader(dataset)
    elif 'multiframes' in model_module_name:
        dataset.X_train = np.array([zip(dataset.X_train, [i]*len(dataset.X_train))
                                    for i in range(FRAMES_PER_VIDEO)]).reshape(-1, 2)
        dataset.X_val = np.array([zip(dataset.X_val, [i]*len(dataset.X_val))
                                  for i in range(FRAMES_PER_VIDEO)]).reshape(-1, 2)
        dataset.X_test = np.array([zip(dataset.X_test, [i]*len(dataset.X_test))
                                   for i in range(FRAMES_PER_VIDEO)]).reshape(-1, 2)
        dataset.y_train = np.tile(np.array(dataset.y_train).reshape(-1), FRAMES_PER_VIDEO).reshape(-1, dataset.n_classes)
        dataset.y_val = np.tile(np.array(dataset.y_val).reshape(-1), FRAMES_PER_VIDEO).reshape(-1, dataset.n_classes)
        dataset.y_test = np.tile(np.array(dataset.y_test).reshape(-1), FRAMES_PER_VIDEO).reshape(-1, dataset.n_classes)
        dataset.data_loader = video.get_loader(dataset)
    else:
        dataset.data_loader = multimodal.get_loader(dataset)
    return dataset


class FCVID(Dataset):

    meta_location = FCVID_META_FILE
    video_aux_info = pickle.load(open(FCVID_INFO_FILE))
    audio_storage = FCVID_AUDIO_STORAGE
    video_storage = FCVID_VIDEO_STORAGE
    spectrogram_storage = audio_storage
    frame_storage = video_storage
    multimodal_features = FCVID_MULTIMODAL_FEATURES
    n_classes = 12
    n_max_frames = 100
    categories = [u'accordionPerformance', u'celloPerformance', u'chamberMusic', u'flutePerformance',
                  u'guitarPerformance', u'harmonicaPerformance', u'pianoPerformance', u'rockBandPerformance',
                  u'saxophonePerformance', u'symphonyOrchestraPerformance', u'trumpetPerformance', u'violinPerformance']

    def get_audio_filename(self, video_id):
        if video_id == 'demo':
            return DEMO_AUDIO

        filename = '{storage}/{video_id}.{ext}'.format(
            storage=self.audio_storage,
            video_id=video_id,
            ext='.wav'
        )
        return filename

    def get_video_filename(self, video_id):
        # Check if it's indexed video frame
        if len(video_id) == 2:
            video_id = video_id[1]
        if video_id == 'demo':
            return DEMO_VIDEO

        filename = '{storage}/{video_id}.{ext}'.format(
            storage=self.video_storage,
            video_id=video_id,
            ext='.avi'
        )
        return filename

    def get_spectrogram_filename(self, video_id):
        return self.get_audio_filename(video_id) + '.spec.npy'

    def get_frames_filename(self, video_id):
        return self.get_video_filename(video_id) + '.frames.lzf.h5'
