Code for paper "[Musical Instrument Recognition in User-generated Videos using a Multimodal Convolutional Neural Network Architecture](http://zenodo.org)" by Olga Slizovskaia, Emilia Gomez, Gloria Haro. ICMR 2017 

We propose an approach to recognize musical instruments in video recordings based on both audio and video information.

# Requirements

numpy
pandas
librosa
hdf5
sklearn
opencv-python
keras>=2.0
tensorflow>=1.0

# Pretrained models

The pretrained models for audio, video and both modalities can be downloaded from [here](http://fcvid) for FCVID dataset and from [here](http://fcvid) for a subset of YouTube-8M dataset. 
models.video.multiframes.datasets.yt8m.hdf5 - 20 frames
models.video.multiframes.datasets.fcvid.hdf5 - 50 frames

# Demo

python launch_experiment.py -t preprocess -m audio.xception -d fcvid -o
python launch_experiment.py -t evaluate -m audio.xception -d fcvid -o

python launch_experiment.py -t preprocess -m video.multiframes -d yt8m -o
python launch_experiment.py -t evaluate -m video.multiframes -d yt8m -o


# Datasets

## Features for multimodal learning

### FCVID

#### Audio features

~/audio_crnn/fcvid_audio_features_crnn.h5 19M
~/audio_experimental/fcvid_audio_features.h5 19M

#### Video features

~/video_deeppool/fcvid_50_frames_video_features.h5 134M

### YouTube-8M

#### Audio features

~/audio_crnn/y8m_audio_features_crnn.h5 223M
~/audio_experimental/y8m_audio_features_crnn.h5 214M

#### Video features

~/video_deeppool/youtube_20_pretrained_1_1_features.h5 1,2G

## Raw video data

### FCVID

Please, find the information about FCVID dataset at the [official webpage](http://bigvid.fudan.edu.cn/FCVID/). From the whole dataset you will be needed only "Music -> Musical performance with instruments" subcategory.
Once downloaded, change the stored path at ```datasets.py``` file.
Additional information: ```~/fcvid_dataset.csv```

### YouTube-8M

The used subset contains most represented musical instruments from YouTube-8M dataset. 
You can find a list of Youtube video ids of videos ```~/y8m_dataset.csv```

# Models

## Audio

CRNN model has been adopted from Choi et. al. original paper and implementation
Xception model has been adapted from Chollet original paper and implementation

## Video

We used InceptionV3 base architecture for video-based experiments. Consider the multi-frame model as a wrapper over Inception V3

## Multimodal

# Reproducing results

