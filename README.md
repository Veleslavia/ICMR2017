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

The pretrained models for audio, video and both modalities can be downloaded from [here](http://update_with_zenodo_url) for FCVID dataset and for a subset of YouTube-8M dataset. 

The following weight files could be found:

`models.video.multiframes.datasets.yt8m.hdf5` has been trained on 20 frames per video
`models.video.multiframes.datasets.fcvid.hdf5` has been trained on 50 frames per video

# Demo

You can try to predict probabilities for the categories using the test option (-o) for the script `launch_experiment.py` which will operate on the demo recording.
For evaluation, please, download the weight files and arrange them at `./weights` folder. 

```
python launch_experiment.py -t preprocess -m audio.xception -d fcvid -o
python launch_experiment.py -t evaluate -m audio.xception -d fcvid -o

python launch_experiment.py -t preprocess -m video.multiframes -d yt8m -o
python launch_experiment.py -t evaluate -m video.multiframes -d yt8m -o

python launch_experiment.py -t evaluate -m multimodal -d yt8m -o
python launch_experiment.py -t evaluate -m multimodal -d fcvid -o
```

# Datasets

## Raw video data

### FCVID

Please, find the information about FCVID dataset at the [official webpage](http://bigvid.fudan.edu.cn/FCVID/). From the whole dataset you will be needed only "Music -> Musical performance with instruments" subcategory.
Once downloaded, change the stored path in ```datasets.py``` file.
The video ids and categories can be found in ```./datasets/fcvid_dataset.csv```

### YouTube-8M

The used subset contains most represented musical instruments from YouTube-8M dataset. 
You can find a list of Youtube video ids of videos in the following file: ```./datasets/y8m_dataset.csv```.

# Models

### Audio

CRNN model has been adopted from Choi et. al. original paper and implementation.
Xception model has been adapted from Chollet original paper and implementation.

### Video

We used InceptionV3 base architecture for video-based experiments. Consider the multi-frame model as a wrapper over Inception V3.

### Multimodal

The multimodal model is working over the features of penultimate layers of audio and video models respectively.
It consist of 2 Dense and 2 BatchNormalization layers. You can find the model at `./models/multimodal.py`.

