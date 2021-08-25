import logging
import warnings
import ctypes
import math
from enum import IntEnum
from typing import Callable, Tuple, List, Dict, Any, Union, Type

import numpy as np

import PythonExtras.numpy_extras as npe
import PythonExtras.patching_tools as patching_tools
import PythonExtras.logging_tools as logging_tools
from PythonExtras.CppWrapper import CppWrapper
from PythonExtras.BufferedNdArray import BufferedNdArray


class MtPatchExtractor:
    """
    A Python wrapper around a C++ class implementing patching of 4D volumes.
    The implementation is derived from the older patching functions and is meant to replace them all,
    implementing support for all their features, including strides, undersampling and multithreading,
    but also patch-to-patch prediction (y-patches) and multivariate data
    (also possibly SDF-based empty space checking in the future).

    Unlike the older implementations, the patch size (now patch-X size) no longer counts in the predicted frame.

    This wrapper stores a pointer to a C++ object and uses ctypes and static wrapper
    functions to call the class methods.
    """

    def __init__(self, volumeData: BufferedNdArray,
                 patchXSize: Tuple, patchYSize: Tuple, patchStride: Tuple, patchInnerStride: Tuple,
                 predictionDelay: int,
                 detectEmptyPatches: bool, emptyValue: float,
                 emptyCheckFeature: int,
                 undersamplingProbAny: float, undersamplingProbEmpty: float, undersamplingProbNonempty: float,
                 inputBufferSize: int, threadNumber: int = 8, logger: logging.Logger = None):

        # self.dtype = volumeData.dtype
        self._volumeData = volumeData
        self._patchXSize = patchXSize
        self._patchYSize = patchYSize
        self._patchStride = patchStride
        self._patchInnerStride = patchInnerStride
        self._predictionDelay = predictionDelay
        self._detectEmptyPatches = detectEmptyPatches
        self._emptyValue = emptyValue
        self._emptyCheckFeature = emptyCheckFeature
        self._undersamplingProbAny = undersamplingProbAny
        self._undersamplingProbEmpty = undersamplingProbEmpty
        self._undersamplingProbNonempty = undersamplingProbNonempty
        self._inputBufferSize = inputBufferSize
        self._threadNumber = threadNumber

        # 4D volume with an optional feature dimension.
        assert 4 <= len(volumeData.shape) <= 5
        noFeatures = len(volumeData.shape) == 4

        # Spatiotemporal size of the volume, disregarding the feature dimension
        self._volumeSize = volumeData.shape if noFeatures else volumeData.shape[:-1]
        self._featureNumber = 1 if noFeatures else volumeData.shape[-1]

        self.patchNumber = patching_tools.compute_patch_number(self._volumeSize, self._patchXSize, self._patchYSize,
                                                               self._patchStride, self._patchInnerStride,
                                                               self._predictionDelay)
        self.patchNumberFlat = npe.multiply(self.patchNumber)
        self.dtype = volumeData.dtype
        self._nextBatchStartIndex = 0

        # These describe statistics about the last batch.
        self.patchesIterated = 0   # Total number of patches in the batch (max possible patches).
        self.patchesChecked = 0    # Patches checked for emptiness (i.e. not skipped due to general undersampling)
        self.patchesEmpty = 0      # Number of empty patches among those checked.
        self.patchesExtracted = 0  # Number of patches extracted (empty and nonempty) after second undersampling.
        self.inputEndReached = False

        self._logger = logging_tools.get_null_logger_if_none(logger)

        self._cPointer = CppWrapper.mtpe_construct(
            self._volumeData,
            self._volumeSize,
            self._featureNumber,
            self._patchXSize,
            self._patchYSize,
            self._patchStride,
            self._patchInnerStride,
            self._predictionDelay,
            self._detectEmptyPatches,
            self._emptyValue,
            self._emptyCheckFeature,
            self._undersamplingProbAny,
            self._undersamplingProbEmpty,
            self._undersamplingProbNonempty,
            self._inputBufferSize,
            self._threadNumber)

        # Defines how to convert this instance when calling C++ function using ctypes.
        self._as_parameter_ = ctypes.c_void_p(self._cPointer)

    def destruct(self):
        CppWrapper.mtpe_destruct(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destruct()

    def has_next_batch(self) -> bool:
        return self._nextBatchStartIndex < self.patchNumberFlat

    def extract_next_batch(self, batchSize: int,
                           outX: np.ndarray, outY: np.ndarray, outIndices: np.ndarray):

        if self._nextBatchStartIndex >= self.patchNumberFlat:
            raise RuntimeError("Finished extracting batches, no further patches exist.")

        nextBatchStartIndex, inputEndReached, self.patchesChecked, self.patchesEmpty, self.patchesExtracted = \
            CppWrapper.mtpe_extract_batch(self, self._nextBatchStartIndex, batchSize, outX, outY, outIndices)

        if inputEndReached and self.inputEndReached:
            self._logger.warning("Ran out of input data twice in a row. Consider increasing the input buffer size.")

        self.inputEndReached = inputEndReached
        self.patchesIterated = nextBatchStartIndex - self._nextBatchStartIndex
        self._nextBatchStartIndex = nextBatchStartIndex

        return self.patchesExtracted

    def get_progress(self) -> float:
        return self._nextBatchStartIndex / self.patchNumberFlat * 100

