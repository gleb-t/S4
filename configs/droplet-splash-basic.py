from typing import *
import os

from PythonExtras import config_tools
from Siamese.config import SiameseConfig
from Siamese.data_types import PatchDesc, RandomSetSampleDesc


def configure(config: SiameseConfig):
    config.dataPath = os.path.expandvars(r'$DEV_VOLUME_DATA_PATH/droplet-splash-itlr/sampled-110-x2-t10/*.dat')

    config.downsampleFactor = 2
    config.volumeCrop = (slice(None), slice(None), slice(None), slice(None))

    config.patchShape = (3, None, 50, 50)
    config.patchMaxOffsetTime = 6
    config.patchMaxOffsetSpace = 10

    # The droplet-splash ensemble is not aligned in time.
    config.alignDifferentMembers = False

    config.shotNumber = 2

    config.dataPointNumber = int(5e5)
    config.augmentDataFlip = True
    config.augmentDataMul = True
    config.augmentPatchesSeparately = False
    config.epochNumber = 100
    config.patience = 3
    config.regularizationCoef = 0.0002

    config.ignoreSameMembersWhenMatching = True
    config.ensembleMemberRepr = SiameseConfig.MemberRepr.last
    config.featureRandomSetSamples = []
    config.featureSupportSets = []
    config.figureConfig.ensembleTfName = 'droplet-splash'

    config.featureSupportSets = [
        [(PatchDesc('025_Hyspin-Hexadecan_0.26mm_viewA', (0, 0, 0, 0)), True)],
    ]

    config.featureRandomSetSamples = [
        RandomSetSampleDesc(100, 'crown', 1, 0),
        RandomSetSampleDesc(100, 'crown', 2, 0),
        RandomSetSampleDesc(100, 'crown', 3, 0),
        RandomSetSampleDesc(100, 'crown', 1, 1),
        RandomSetSampleDesc(100, 'crown', 2, 2),
        RandomSetSampleDesc(100, 'crown', 3, 3),

        RandomSetSampleDesc(100, 'splash', 1, 0),
        RandomSetSampleDesc(100, 'splash', 2, 0),
        RandomSetSampleDesc(100, 'splash', 3, 0),
        RandomSetSampleDesc(100, 'splash', 1, 1),
        RandomSetSampleDesc(100, 'splash', 2, 2),
        RandomSetSampleDesc(100, 'splash', 3, 3),
    ]
    config.featureRandomSetSpatialCoords = (0, 12, 60)
    config.featureRandomSetSpatialCoordsMaxOffset = (0, 12, 12)

