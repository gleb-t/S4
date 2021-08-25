from typing import Tuple

from Siamese.config import SiameseConfig


# noinspection DuplicatedCode
def build_encoder(config: SiameseConfig, inputShape: Tuple[int, ...]) -> 'keras.models.Sequential':
    import keras
    from Siamese import layers

    assert inputShape[1] > 1  # We expect a 3D volume.

    reg = keras.regularizers.l2(config.regularizationCoef)

    encoder = keras.models.Sequential()
    # We don't need to change the shape, but we'll keep the layer to specify the input size.
    encoder.add(keras.layers.Reshape(inputShape, input_shape=inputShape))

    encoder.add(layers.Conv4d(64, (1, 3, 3, 3), strides=(1, 2, 2, 2), activation='relu', kernel_regularizer=reg))
    encoder.add(layers.Conv4d(128, (1, 3, 3, 3), strides=(1, 1, 1, 1), activation='relu', kernel_regularizer=reg))
    encoder.add(layers.Conv4d(128, (3, 3, 3, 3), strides=(2, 1, 2, 2), activation='relu', kernel_regularizer=reg))
    encoder.add(layers.Conv4d(256, (1, 3, 3, 3), strides=(1, 2, 2, 2), activation='relu', kernel_regularizer=reg))
    encoder.add(keras.layers.Flatten())
    encoder.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=reg))
    encoder.add(keras.layers.Dense(256, activation='sigmoid', kernel_regularizer=reg, name='encoder_out'))

    return encoder
