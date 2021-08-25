from typing import *
import os

from PythonExtras import config_tools
from Siamese.config import SiameseConfig
from Siamese.data_types import PatchDesc, RandomSetSampleDesc


def configure(config: SiameseConfig):

    configFunc = config_tools.load_subconfig_func(__file__, '200414_cylinder-300-2-basic-paper.py', 'configure')
    configFunc(config)

    config.batchSize = 2
    config.batchSizePredict = 2
    config.dataPointNumber = int(1e2)
    config.epochNumber = 2

