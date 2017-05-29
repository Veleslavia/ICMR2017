from keras.layers import Dense, BatchNormalization
from keras.layers import InputLayer, Flatten
from keras.models import Sequential

from settings import FRAMES_PER_VIDEO

def build_model(n_classes):

    model = Sequential()

    input_shape = (FRAMES_PER_VIDEO*2+1, 1024)
    model.add(InputLayer(input_shape=input_shape, name='input'))
    model.add(Flatten(name='flatten'))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu', name='fc1'))
    model.add(BatchNormalization())
    model.add(Dense(n_classes, activation='softmax', name='predictions'))

    return model
