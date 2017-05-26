import os
import numpy as np
import librosa


def get_loader(dataset):
    def load_audio(video_ids):
        spectrogram_func = _spectrogram_crnn
        shape = (-1, 96, 1366, 1)
        spectrograms = list()
        for youtube_id in video_ids:
            spec_filename = dataset.get_spectrogram_filename(youtube_id)
            if os.path.exists(spec_filename):
                spectrograms.append(np.load(spec_filename))
            else:
                audio_filename = dataset.get_audio_filename(youtube_id)
                spectrogram = spectrogram_func(audio_filename)
                np.save(spec_filename, spectrogram)
                spectrograms.append(spectrogram)
        return np.array(spectrograms).reshape(shape)
    return load_audio


def _spectrogram_crnn(audio_filename):
    return _segments_with_crop_spec(audio_filename)


def _segments_with_crop_spec(filename):
    # mel-spectrogram parameters
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12

    src, sr = librosa.load(filename, sr=SR)
    n_sample = src.shape[0]
    n_sample_wanted = int(DURA * SR)
    n_segments = 10
    n_sample_wanted_per_segment = int(DURA * SR) / 10

    # segment with central crop selection
    if n_sample < n_sample_wanted:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
    elif n_sample > n_sample_wanted:  # if too long
        src = np.hstack((src, np.zeros((n_segments - (n_sample % n_segments),))))
        src = src.reshape(n_segments, -1)
        n_sample_per_segment = src.shape[1]
        croped_src = np.zeros((n_segments, n_sample_wanted_per_segment))
        for i, segment in enumerate(src):
            croped_src[i] = segment[(n_sample_per_segment - n_sample_wanted_per_segment) / 2:
                                    (n_sample_per_segment + n_sample_wanted_per_segment) / 2]
        src = croped_src.reshape(-1)
    logam = librosa.logamplitude
    melgram = librosa.feature.melspectrogram
    x = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                      n_fft=N_FFT, n_mels=N_MELS) ** 2,
              ref_power=1.0)
    return x
