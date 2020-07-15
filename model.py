'''
Author: Dominik Waibel
Adapted from: https://github.com/sayakpaul/SimCLR-in-TensorFlow-2
'''


from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from ResNet50 import ResNet50

def get_resnet_simclr(hidden_1, hidden_2, hidden_3, Image_dimensions):
#   base_model = ResNet50(include_top=False, weights=None, input_shape=Image_dimensions)
    base_model = ResNet50(Image_dimensions,
                     Dropout=0,
                     include_top=True,
                     weights=None,
                     input_tensor=None,
                     pooling='max',
                     classes=100)
    base_model.trainabe = True
    inputs = Input(Image_dimensions)
    h = base_model(inputs, training=True)
    #h = base_model.layers[-1].output
    #h = GlobalAveragePooling2D()(h)
    base_model.summary()
    projection_1 = Dense(hidden_1)(h)
    projection_1 = Activation("relu")(projection_1)
    projection_2 = Dense(hidden_2)(projection_1)
    projection_2 = Activation("relu")(projection_2)
    projection_3 = Dense(hidden_3)(projection_2)

    resnet_simclr = Model(inputs, projection_3)

    return resnet_simclr