from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import AveragePooling2D
from .inception_v3 import InceptionV3


def build_model(n_classes, pretraining=True):

    weights = None if pretraining else 'imagenet'
    input_shape = (224, 224, 3)
    frame_input = Input(shape=input_shape, name='input')
    frame_model = InceptionV3(include_top=False, weights=weights, input_tensor=frame_input)
    x = AveragePooling2D((5, 5), strides=(5, 5), name='av_pool')(frame_model.outputs[0])
    x = Flatten(name='flatten')(x)
    if pretraining:
        frame_model.load_weights('./weights/inceptionv3.pretrained.hdf5',
                                 by_name=True)
    x = Dense(n_classes, activation='softmax', name='predictions')(x)
    model = Model(frame_input, x)

    return model