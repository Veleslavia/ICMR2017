from keras.layers import Dense, BatchNormalization
from keras.layers import InputLayer, Flatten
from keras.models import Model

from settings import FRAMES_PER_VIDEO

def build_model(n_classes):

    input_shape = (FRAMES_PER_VIDEO*2+1, 1024)
    input_data = InputLayer(input_shape=input_shape, name='input')

    #x = Flatten(name='flatten')(input_data)
    x = BatchNormalization()(input_data)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = BatchNormalization()(x)
    x = Dense(n_classes, activation='softmax', name='predictions')(x)

    model = Model(input_data, x)
    return model
