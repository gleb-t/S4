from typing import *
import os

from PythonExtras import config_tools
from Siamese.config import SiameseConfig
from Siamese.data_types import PatchDesc, RandomSetSampleDesc


# noinspection DuplicatedCode
def configure(config: SiameseConfig):
    configFunc = config_tools.load_subconfig_func(__file__, '200414_cylinder-300-2-basic-paper.py', 'configure')
    configFunc(config)

    config.simMetricNames = ['model', 'mse', 'vgg', 'wasserstein', 'ssim', 'hist-l1', 'hist-wasserstein']

    config.checkpointToLoad = os.path.join(os.environ['DEV_OUT_PATH'],
                                           '20200817-153710_cluster-1125_200817_cylinder-300-2-basic_all-metrics',
                                           '200817-153723-crimson-pond_200817_cylinder-300-2-basic_all-metrics',
                                           'model.hdf'
                                           )
    # Compatibility mode for older models.
    config.backCompDropAttr = True

    # Do not train further, use the pretrained model above.
    config.epochNumber = 0
    # Do not generate much training data. (Saves time)
    config.dataPointNumber = int(1e3)
