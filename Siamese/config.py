import os
from typing import *
from enum import Enum

import pyplant
import numpy as np

from PythonExtras import config_tools
from Siamese.data_types import *


if TYPE_CHECKING:
    import keras


class SiameseConfig(pyplant.ConfigBase):

    class FigureConfig:

        def __init__(self):
            self.matchesToShow = 100
            self.margin = 10
            self.ensembleTfName = 'cylinder'
            self.blockWidth, self.blockHeight = 30, 30
            self.groupPadding = 30
            self.iconSize = 10
            self.impostorPadding = 2

    # Deprecated.
    class MemberRepr(Enum):
        unknown = 0
        middle = 1
        last = 2

    def __init__(self):
        super().__init__()

        self.dataPath = ''
        self.featureMetadataPath = ''  # A Json file describing which members have which features.

        self.outputDirPath = ''
        self.tensorboardDirPath = ''
        self.checkpointDirPath = ''
        self.runId = ''

        self.downsampleFactor = 1  # Spatial downsampling factor.
        self.volumeCrop = None  # type: Optional[Tuple[Optional[slice], ...]]
        # Provide 'None' along a dimension to set it equal to the volume shape.
        self.patchShape = (3, None, 41, 41)
        self.patchMaxOffsetTime = 9
        self.patchMaxOffsetSpace = 40

        # The number of "shots" provided for recognizing a feature. A.k.a. support set size.
        self.shotNumber = 3

        # If true, will try to sample the left and right "shots" nearby in space-time,
        # even for negative data points.
        self.alignDifferentMembers = False
        # Whether the left shots should come from a shared neighborhood
        # even for negative data points. (Choose random location, sample around.)
        self.groupLeftShots = True

        self.augmentDataFlip = True  # Whether to do data augmentation by flipping the patches.
        self.augmentDataMul = False  # Whether to do data augmentation by multiplying the patch values.
        self.augmentPatchesSeparately = False  # Whether augment the 'left' patches separately (if true) or together.

        self.checkpointToLoad = None  # type: Optional[str]
        self.backCompDropAttr = False  # Drop the attribute dim. Useful for reproducing results with older models.

        # self.encoderCtor = config_tools.load_subconfig_func(__file__, 'models/encoder-basic.py', 'build_encoder')
        self.encoderCtor = build_encoder
        self.classifierCtor = build_classifier
        self.vggPretrainedLayerName = 'fc1'  # Which layer to use as encoding for the pretrained VGG similarity.
        self.vggTunedLayerName = 'fc1'  # Which layer to use as encoding for the tuned VGG encoder.
        self.vggTunedMode = None  # How to tune VGG. (last/all)
        self.normalizerCoefs = None  # type: Optional[List[List[int]]]  # Needed for finetuning models.

        self.dataPointNumber = int(1e5)
        self.testSplitRatio = 0.1
        self.epochNumber = 10
        self.patience = 3
        self.regularizationCoef = 0.001
        self.learningRate = 0.0001
        self.batchSize = 32
        self.batchSizePredict = 128
        self.checkpointIntervalMinutes = 30
        self.dtypeModel = np.float32
        self.useCustomTrainer = True   # Use the custom macrobatched trainer or keras.fit ?
        self.trainMacrobatchSizeBytes = int(4e9)

        self.matchPatchStrideFactor = 1.0
        self.matchBatchSizeBytes = int(2e9)
        # Deprecated. We now always compute both versions of the ranking metrics.
        self.ignoreSameMembersWhenMatching = False
        # Deprecated. We no longer compute ensemble distances during the run.
        self.ensembleMemberRepr = SiameseConfig.MemberRepr.last

        self.simMetricNames = ['model', 'mse']
        self.aggFuncNames = ['mean']
        # self.featureAggNames = ['mean', 'min']
        
        self.metricHistBinNumber = 16

        self.featureSupportSets = [
            [(PatchDesc('memberA', (5, 0, 220, 40)), True)],
            [(PatchDesc('memberA', (5, 0, 220, 40)), True),
             (PatchDesc('memberB', (5, 0, 220, 40)), True)],
        ]  # type: List[TSupportSetPatchList]
        self.featureRandomSetSeed = 6987191424553
        self.featureRandomSetSamples = [
            RandomSetSampleDesc(50, 't', 5, 5),
            RandomSetSampleDesc(50, 'l', 5, 5),
        ]  # type: List[RandomSetSampleDesc]
        self.featureRandomSetSpatialCoords = None  # type: Optional[Tuple[int, ...]]
        self.featureRandomSetSpatialCoordsMaxOffset = None  # type: Optional[Tuple[int, ...]]

        self.figureConfig = SiameseConfig.FigureConfig()

        self.loggerName = 'siamese'

        self.mark_auxiliary(['outputDirPath', 'tensorboardDirPath', 'checkpointDirPath', 'runId', 'loggerName'])


# noinspection DuplicatedCode
def build_encoder(config: SiameseConfig, inputShape: TupleInt) -> 'keras.models.Sequential':
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


def build_encoder_3d(config: SiameseConfig, inputShape: TupleInt) -> 'keras.models.Sequential':
    """
    An example implementation of a 3D+T patch encoder, this one isn't used anywhere,
    since encoders are defined in the configuration files.
    """
    import keras
    from Siamese import layers

    assert inputShape[1] > 1  # We expect a 3D volume.

    reg = keras.regularizers.l2(config.regularizationCoef)

    encoder = keras.models.Sequential()
    # We don't need to change the shape, but we'll keep the layer to specify the input size.
    encoder.add(keras.layers.Reshape(inputShape, input_shape=inputShape))

    encoder.add(layers.ConvNd(64, (1, 3, 3, 3), strides=(1, 2, 2, 2), activation='relu', kernel_regularizer=reg))
    encoder.add(layers.ConvNd(128, (1, 3, 3, 3), strides=(1, 1, 1, 1), activation='relu', kernel_regularizer=reg))
    encoder.add(layers.ConvNd(128, (3, 3, 3, 3), strides=(2, 2, 2, 2), activation='relu', kernel_regularizer=reg))
    encoder.add(layers.ConvNd(256, (1, 3, 3, 3), strides=(1, 2, 2, 2), activation='relu', kernel_regularizer=reg))
    encoder.add(keras.layers.Flatten())
    encoder.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=reg))
    encoder.add(keras.layers.Dense(256, activation='sigmoid', kernel_regularizer=reg, name='encoder_out'))

    return encoder


def build_classifier(config: SiameseConfig, inputShape: Tuple[int, ...]) -> 'keras.models.Sequential':
    import keras

    classifier = keras.models.Sequential()
    classifier.add(keras.layers.Dense(1, activation='sigmoid', input_shape=inputShape))

    return classifier
