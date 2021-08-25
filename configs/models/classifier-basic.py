from typing import *

from Siamese.config import SiameseConfig


# noinspection DuplicatedCode
def build_classifier(config: SiameseConfig, inputShape: Tuple[int, ...]) -> 'keras.models.Sequential':
    import keras

    classifier = keras.models.Sequential()
    classifier.add(keras.layers.Dense(1, activation='sigmoid', input_shape=inputShape))

    return classifier
