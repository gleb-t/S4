from typing import Tuple, TYPE_CHECKING

from Siamese.config import SiameseConfig
from Siamese.layers import VggAdapterLayer

if TYPE_CHECKING:
    import keras


# noinspection DuplicatedCode
def build_encoder(config: SiameseConfig, inputShape: Tuple[int, ...]) -> 'keras.models.Sequential':
    import keras
    from keras.applications.vgg16 import VGG16

    fullModel = VGG16(weights='imagenet', include_top=True)
    vggModel = keras.models.Model(inputs=fullModel.input, outputs=fullModel.get_layer(config.vggTunedLayerName).output)

    if config.vggTunedMode == 'all':
        pass
    elif config.vggTunedMode == 'last':
        for layer in vggModel.layers[:-1]:
            layer.trainable = False
    else:
        raise ValueError("Unknown tuning mode: {}".format(config.vggTunedMode))

    encoder = keras.models.Sequential()
    # Pulled all the transforms into a custom layer to enable serialization.
    # Drop unneeded axes, take the first frame, resize the image.

    # The input shape will be changed by vgg_preprocess (an ugly work-around).
    inputShape = (*inputShape[2:4], 3)
    encoder.add(VggAdapterLayer(input_shape=inputShape))
    for layer in vggModel.layers:
        encoder.add(layer)

    # Name the last layer.
    encoder.layers[-1].name = 'encoder_out'

    return encoder
