STORAGE = '/storage/olga'

TEST_VAL_SIZE = 0.3

FCVID_AUDIO_STORAGE = ''
FCVID_VIDEO_STORAGE = ''
FCVID_AUDIO_FEATURE_FILE = '/home/olga/audio_crnn/fcvid_audio_features_crnn.h5'
FCVID_VIDEO_FEATURE_FILE = '/home/olga/video_deeppool/fcvid_50_pretrained_features.h5'
FCVID_INFO_FILE = './datasets/fcvid_auxinfo.pkl'
FCVID_META_FILE = './datasets/fcvid_dataset.csv'

YT8M_AUDIO_STORAGE = ''
YT8M_VIDEO_STORAGE = '/home/olga/y8m'
YT8M_AUDIO_FEATURE_FILE = '/home/olga/audio_crnn/y8m_audio_features_crnn.h5'
YT8M_VIDEO_FEATURE_FILE = '/home/olga/video_deeppool/youtube_20_pretrained_1_1_features.h5'
YT8M_INFO_FILE = './datasets/yt8m_auxinfo.pkl'
YT8M_META_FILE = './datasets/yt8m_dataset.csv'

DEMO_STORAGE = './demo'
DEMO_AUDIO = './demo/demo.wav'
DEMO_VIDEO = './demo/demo.mp4'

MULTIMODAL_BATCHSIZE = 32
UNIMODAL_BATCHSIZE = 16
BATCHSIZE = 16
PATIENCE = 10
NUM_EPOCHS = 200
LR_REDUCE_EVERY_K_EPOCH = 10
INIT_LR = 0.001

FRAMES_PER_VIDEO = 20

CAP_PROP_POS_MSEC = 0