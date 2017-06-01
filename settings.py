STORAGE = '/storage/olga'

TEST_VAL_SIZE = 0.3

FCVID_AUDIO_STORAGE = ''
FCVID_VIDEO_STORAGE = ''
FCVID_MULTIMODAL_FEATURES = './features/fcvid.features.h5'
FCVID_INFO_FILE = './datasets/fcvid_auxinfo.pkl'
FCVID_META_FILE = './datasets/fcvid_dataset.csv'

YT8M_AUDIO_STORAGE = ''
YT8M_VIDEO_STORAGE = ''
YT8M_MULTIMODAL_FEATURES = './features/yt8m.features.h5'
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