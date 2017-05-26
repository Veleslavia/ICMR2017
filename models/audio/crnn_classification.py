from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, BatchNormalization
from keras.layers.advanced_activations import ELU


def build_model(n_classes, dense=True):
# Adapted from Kenwoo Choi original code

    input_shape = (96, 1366, 1)
    melgram_input = Input(shape=input_shape)
    channel_axis = 3
    freq_axis = 1
    time_axis = 2

    # Input block
    x = ZeroPadding2D(padding=(0, 37))(melgram_input)
    x = BatchNormalization(axis=time_axis, name='bn_0_freq')(x)

    # Conv block 1
    x = Convolution2D(64, (3, 3), padding='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)

    # Conv block 2
    x = Convolution2D(128, (3, 3), padding='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2')(x)

    # Conv block 3
    x = Convolution2D(128, (3, 3), padding='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3')(x)

    # Conv block 4
    x = Convolution2D(128, (3, 3), padding='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(1024, name='fc1')(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    if dense:
        x = Dense(n_classes, activation='softmax', name='prediction')(x)

    model = Model(melgram_input, x)
    return model
