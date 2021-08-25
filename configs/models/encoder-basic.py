from typing import Tuple

from Siamese.config import SiameseConfig


# noinspection DuplicatedCode
def build_encoder(config: SiameseConfig, inputShape: Tuple[int, ...]) -> 'keras.models.Sequential':
    import keras

    reg = keras.regularizers.l2(config.regularizationCoef)

    encoder = keras.models.Sequential()
    # For 2D+T data we skip the Z dimension and use 3D convolutions.
    encoder.add(keras.layers.Reshape((inputShape[0], *inputShape[2:]), input_shape=inputShape))

    encoder.add(keras.layers.Conv3D(64, (1, 3, 3), strides=(1, 2, 2), activation='relu', kernel_regularizer=reg))
    encoder.add(keras.layers.Conv3D(128, (1, 3, 3), strides=(1, 1, 1), activation='relu', kernel_regularizer=reg))
    encoder.add(keras.layers.Conv3D(128, (3, 3, 3), strides=(2, 2, 2), activation='relu', kernel_regularizer=reg))
    encoder.add(keras.layers.Conv3D(256, (1, 3, 3), strides=(1, 2, 2), activation='relu', kernel_regularizer=reg))
    encoder.add(keras.layers.Flatten())
    encoder.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=reg))
    encoder.add(keras.layers.Dense(256, activation='sigmoid', kernel_regularizer=reg, name='encoder_out'))

    return encoder
