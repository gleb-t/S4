import logging
import os
from typing import *

import numpy as np

from PythonExtras import logging_tools
from PythonExtras import volume_tools
from PythonExtras import numpy_extras as npe
from PythonExtras.BufferedNdArray import BufferedNdArray
from PythonExtras.MtPatchExtractor import MtPatchExtractor


def _build_volume_data_provider(volumePaths: List[str],
                                logger: Optional[logging.Logger] = None) -> Iterable[npe.LargeArray]:
    logger = logging_tools.get_null_logger_if_none(logger)

    def _load_volume_into_bna(volumePath: str) -> npe.LargeArray:
        return volume_tools.load_volume_data_from_dat(
            volumePath,
            outputAllocator=volume_tools.allocate_temp_bna,
            forceMultivar=True,
            printFn=lambda s: logger.debug(s, extra={'throttlingId': 'load_volume_data_from_dat'})
        )

    # return (_load_volume_into_bna(p) for p in volumePaths)
    for i, p in enumerate(volumePaths):
        logger.info("Preparing to extract patches from dataset '{}' ({}/{})."
                    .format(os.path.basename(p), i + 1, len(volumePaths)))
        yield _load_volume_into_bna(p)


def _call_exit(obj: object):
    if hasattr(obj, '__exit__'):
        obj.__exit__(None, None, None)


class MultidataPatchExtractor:
    """
    A quick and dirty class mimicking the interface of MtPatchProvider.
    Instead of taking volume data, it accepts a list of volume paths,
    loads them and returns a stream of patches from all of them,
    instantiating MtPatchProviders as needed.
    """

    def __init__(self,
                 volumes: Union[List[str], Iterable[npe.LargeArray]],
                 patchXSize: Tuple,
                 patchYSize: Tuple,
                 patchStride: Tuple,
                 patchInnerStride: Tuple,
                 predictionDelay: int,
                 detectEmptyPatches: bool,
                 emptyValue: float,
                 emptyCheckFeature: int,
                 undersamplingProbAny: float,
                 undersamplingProbEmpty: float,
                 undersamplingProbNonempty: float,
                 inputBufferSize: int,
                 threadNumber: int = 8,
                 logger: logging.Logger = None):

        # Handle both a list of paths and iterables of volumes.
        if isinstance(volumes, list) and isinstance(volumes[0], str):
            self._volumesIterable = _build_volume_data_provider(volumes, logger)
            self._volumeNumber = len(volumes)
        else:
            self._volumesIterable = volumes
            self._volumeNumber = None  # We don't know how many volumes will be loaded.

        self._volumeIterator = iter(self._volumesIterable)

        # These members are identical to an MtPatchExtractor (copy-paste).

        # self.dtype = volumeData.dtype
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

        # These describe statistics about the last batch.
        self.patchesIterated = 0   # Total number of patches in the batch (max possible patches).
        self.patchesChecked = 0    # Patches checked for emptiness (i.e. not skipped due to general undersampling)
        self.patchesEmpty = 0      # Number of empty patches among those checked.
        self.patchesExtracted = 0  # Number of patches extracted (empty and nonempty) after second undersampling.
        self.inputEndReached = False
        self.datasetIndex = -1

        # Members needed to support the multi-dataset data loading.
        self._activeVolumeData = None  # type: Optional[BufferedNdArray]
        self._activePatchExtractor = None  # type: Optional[MtPatchExtractor]
        self._activeDatasetIndex = 0
        self._isFinished = False

        # Tracking additional stats.
        self.patchNumberFlat = 0

        self._logger = logging_tools.get_null_logger_if_none(logger)

    def destruct(self):
        if self._activePatchExtractor is not None:
            self._activePatchExtractor.destruct()
            _call_exit(self._activeVolumeData)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destruct()

    def has_next_batch(self) -> bool:
        # todo
        # activeProviderReady = self._activePatchExtractor is None or self._activePatchExtractor.has_next_batch()
        return not self._isFinished  # and (activeProviderReady)

    def extract_next_batch(self, batchSize: int,
                           outX: np.ndarray, outY: np.ndarray, outIndices: np.ndarray):

        if self._isFinished:
            raise RuntimeError("Finished extracting batches, no further datasets exist.")

        if self._activePatchExtractor is None:
            self._try_load_next_volume()

        # Do the actual patch extraction.
        self._activePatchExtractor.extract_next_batch(batchSize, outX, outY, outIndices)

        # Copy over the last batch statistics, for interface compatibility.
        self.patchesIterated, self.patchesChecked, self.patchesEmpty, self.patchesExtracted, self.inputEndReached = \
            self._activePatchExtractor.patchesIterated, \
            self._activePatchExtractor.patchesChecked, \
            self._activePatchExtractor.patchesEmpty, \
            self._activePatchExtractor.patchesExtracted, \
            self._activePatchExtractor.inputEndReached

        # Also store from which dataset the data came. (Can be read by the client code.)
        self.datasetIndex = self._activeDatasetIndex

        # Destroy the extractor and the temp data when we're done.
        if not self._activePatchExtractor.has_next_batch():
            self._activePatchExtractor.destruct()
            self._activePatchExtractor = None
            _call_exit(self._activeVolumeData)
            self._activeVolumeData = None

            self._logger.debug("Finished extracting patches from the current dataset.")
            self._activeDatasetIndex += 1

            # Try to load the next volume, so that we know if the next batch exists.
            try:
                self._try_load_next_volume()
            except StopIteration:
                self._isFinished = True
                return self.patchesExtracted

        return self.patchesExtracted

    def get_progress(self, volumeNumber: Optional[int] = None) -> float:
        volumeNumber = volumeNumber or self._volumeNumber
        # Sometimes we might not know how many volumes will be loaded in total.
        if volumeNumber is None:
            return 0.0

        progress = self._activeDatasetIndex / volumeNumber * 100
        if self._activePatchExtractor is not None:
            progress += self._activePatchExtractor.get_progress() / volumeNumber

        return progress

    def _try_load_next_volume(self):

        volumeData = next(self._volumeIterator)

        # todo This is a pretty dirty way of making the patch provider work with non-bna data.
        #      Ideally, we would be able to handle data in RAM with no overhead of creating temp files.
        if not isinstance(volumeData, BufferedNdArray):
            self._activeVolumeData = volume_tools.allocate_temp_bna(volumeData.shape, volumeData.dtype)
            self._activeVolumeData[...] = volumeData
        else:
            self._activeVolumeData = volumeData

        # Initialize an extractor to read the patches from it.
        self._activePatchExtractor = MtPatchExtractor(
            self._activeVolumeData,
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
            self._threadNumber,
            self._logger
        )

        self.patchNumberFlat += self._activePatchExtractor.patchNumberFlat

