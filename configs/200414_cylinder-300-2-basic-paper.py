from typing import *
import os

from PythonExtras import config_tools
from Siamese.config import SiameseConfig
from Siamese.data_types import PatchDesc, RandomSetSampleDesc


# noinspection DuplicatedCode
def configure(config: SiameseConfig):
    config.dataPath = os.path.expandvars(r'$DEV_VOLUME_DATA_PATH/cylinder-ensemble/sampled-300-2/*.dat')

    config.downsampleFactor = 2
    config.volumeCrop = (slice(15, None), slice(0, 1), slice(220, None), slice(None))
    config.patchShape = (3, None, None, None)
    config.patchMaxOffsetTime = 6
    config.patchMaxOffsetSpace = 10

    # The cylinder ensemble is time and space aligned, so we can enable this.
    config.alignDifferentMembers = True

    config.shotNumber = 2
    config.dataPointNumber = int(5e5)
    config.augmentDataFlip = True
    config.augmentDataMul = True
    config.augmentPatchesSeparately = False
    config.epochNumber = 100
    config.patience = 3
    config.regularizationCoef = 0.0002

    config.encoderCtor = config_tools.load_subconfig_func(__file__, 'models/encoder-basic.py', 'build_encoder')

    # config.checkpointToLoad = r'some/path/to/a/keras/checkpoint.hdf'
    # config.epochNumber = 0

    config.ignoreSameMembersWhenMatching = True
    config.simMetricNames = ['model', 'mse']

    config.ensembleMemberRepr = SiameseConfig.MemberRepr.last
    config.featureRandomSetSamples = [
        RandomSetSampleDesc(100, 't', 1, 0),
        RandomSetSampleDesc(100, 't', 2, 0),
        RandomSetSampleDesc(100, 't', 3, 0),
        RandomSetSampleDesc(100, 't', 1, 1),
        RandomSetSampleDesc(100, 't', 2, 2),
        RandomSetSampleDesc(100, 't', 3, 3),

        RandomSetSampleDesc(100, 'l', 1, 0),
        RandomSetSampleDesc(100, 'l', 2, 0),
        RandomSetSampleDesc(100, 'l', 3, 0),
        RandomSetSampleDesc(100, 'l', 1, 1),
        RandomSetSampleDesc(100, 'l', 2, 2),
        RandomSetSampleDesc(100, 'l', 3, 3),
    ]
    config.featureSupportSets = [
        [(PatchDesc('cylinder_100_20_70', (45, 0, 220, 0)), True)],
        [(PatchDesc('cylinder_100_20_70', (45, 0, 220, 0)), True),
         (PatchDesc('cylinder_70_20_50', (45, 0, 220, 0)), True)],
        [(PatchDesc('cylinder_100_20_70', (45, 0, 220, 0)), True),
         (PatchDesc('cylinder_70_20_50', (45, 0, 220, 0)), True),
         (PatchDesc('cylinder_70_40_40', (45, 0, 220, 0)), True)],

        [(PatchDesc('cylinder_100_20_70', (45, 0, 220, 0)), True),
         (PatchDesc('cylinder_100_5_80', (25, 0, 220, 0)), False)],
        [(PatchDesc('cylinder_100_20_70', (45, 0, 220, 0)), True),
         (PatchDesc('cylinder_70_20_50', (45, 0, 220, 0)), True),
         (PatchDesc('cylinder_100_5_80', (25, 0, 220, 0)), False),
         (PatchDesc('cylinder_40_0_60', (15, 0, 220, 0)), False)],
        [(PatchDesc('cylinder_100_20_70', (45, 0, 220, 0)), True),
         (PatchDesc('cylinder_70_20_50', (45, 0, 220, 0)), True),
         (PatchDesc('cylinder_70_40_40', (45, 0, 220, 0)), True),
         (PatchDesc('cylinder_100_5_80', (25, 0, 220, 0)), False),
         (PatchDesc('cylinder_40_0_60', (15, 0, 220, 0)), False),
         (PatchDesc('cylinder_70_0_70', (35, 0, 220, 0)), False)],

        [(PatchDesc('cylinder_80_25_30', (40, 0, 220, 0)), True)],
        [(PatchDesc('cylinder_80_25_30', (40, 0, 220, 0)), True),
         (PatchDesc('cylinder_70_20_50', (45, 0, 220, 0)), True)],
        [(PatchDesc('cylinder_80_25_30', (40, 0, 220, 0)), True),
         (PatchDesc('cylinder_70_20_50', (45, 0, 220, 0)), True),
         (PatchDesc('cylinder_90_40_60', (30, 0, 220, 0)), True)],

        [(PatchDesc('cylinder_80_25_30', (40, 0, 220, 0)), True),
         (PatchDesc('cylinder_40_35_30', (25, 0, 220, 0)), False)],
        [(PatchDesc('cylinder_80_25_30', (40, 0, 220, 0)), True),
         (PatchDesc('cylinder_70_20_50', (45, 0, 220, 0)), True),
         (PatchDesc('cylinder_40_35_30', (25, 0, 220, 0)), False),
         (PatchDesc('cylinder_70_0_70', (15, 0, 220, 0)), False)],
        [(PatchDesc('cylinder_80_25_30', (40, 0, 220, 0)), True),
         (PatchDesc('cylinder_70_20_50', (45, 0, 220, 0)), True),
         (PatchDesc('cylinder_90_40_60', (30, 0, 220, 0)), True),
         (PatchDesc('cylinder_40_35_30', (25, 0, 220, 0)), False),
         (PatchDesc('cylinder_70_0_70', (15, 0, 220, 0)), False),
         (PatchDesc('cylinder_50_40_60', (45, 0, 220, 0)), False)],

        [(PatchDesc('cylinder_50_20_70', (45, 0, 220, 0)), True)],
        [(PatchDesc('cylinder_50_20_70', (45, 0, 220, 0)), True),
         (PatchDesc('cylinder_40_0_80', (45, 0, 220, 0)), True)],
        [(PatchDesc('cylinder_50_20_70', (45, 0, 220, 0)), True),
         (PatchDesc('cylinder_40_0_80', (45, 0, 220, 0)), True),
         (PatchDesc('cylinder_50_15_50', (45, 0, 220, 0)), True)],

        [(PatchDesc('cylinder_50_20_70', (45, 0, 220, 0)), True),
         (PatchDesc('cylinder_90_40_60', (30, 0, 220, 0)), False)],
        [(PatchDesc('cylinder_50_20_70', (45, 0, 220, 0)), True),
         (PatchDesc('cylinder_40_0_80', (45, 0, 220, 0)), True),
         (PatchDesc('cylinder_90_40_60', (30, 0, 220, 0)), False),
         (PatchDesc('cylinder_80_25_30', (45, 0, 220, 0)), False)],
        [(PatchDesc('cylinder_50_20_70', (45, 0, 220, 0)), True),
         (PatchDesc('cylinder_40_0_80', (45, 0, 220, 0)), True),
         (PatchDesc('cylinder_50_15_50', (45, 0, 220, 0)), True),
         (PatchDesc('cylinder_90_40_60', (30, 0, 220, 0)), False),
         (PatchDesc('cylinder_80_25_30', (45, 0, 220, 0)), False),
         (PatchDesc('cylinder_70_35_70', (45, 0, 220, 0)), False)],
    ]

    config.figureConfig.ensembleTfName = 'cylinder-300'
    config.figureConfig.blockHeight = 3 * config.figureConfig.blockWidth
