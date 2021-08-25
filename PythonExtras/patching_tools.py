import logging
import math
import time
import warnings
from typing import Tuple, Union

import h5py
import numpy as np
from deprecation import deprecated

from PythonExtras import numpy_extras as npe
from PythonExtras.CppWrapper import CppWrapper


def validate_input_buffer_size(volumeShape: Tuple, patchXSize: Tuple, patchYSize: Tuple, patchInnerStride: Tuple,
                               predictionDelay: int, dtype: np.dtype, proposedBufferSize: int):
    """
    Computes the minumum buffer size for an input volume, such that at least some patches can be extracted.
    """
    patchSupportFirstAxis = (patchXSize[0] - 1) * patchInnerStride[0] + patchYSize[0] + predictionDelay
    requiredBufferSize = npe.multiply(volumeShape[1:]) * patchSupportFirstAxis * dtype.itemsize
    if requiredBufferSize > proposedBufferSize:
        warnings.warn("Increased the proposed input buffer size {:,} to {:,}"
                      .format(proposedBufferSize, requiredBufferSize))

    return max(requiredBufferSize, proposedBufferSize)


def extract_patches_slow(data: Union[np.ndarray, h5py.Dataset], sourceAxes: Tuple[int], patchSize: Tuple,
                         patchStride: Tuple = None, verbose=False):
    """
    The same method, but it's old non-Cpp version.

    :param data:
    :param sourceAxes:
    :param patchSize:
    :param patchStride:
    :param verbose:
    :return:
    """
    # todo Implement C++ support for different dtypes.

    # By default, extract every patch.
    if patchStride is None:
        patchStride = (1,) * len(sourceAxes)

    result = None

    patchCenters = []  # Store geometrical center of each path. (Can be useful for the caller.)
    patchNumber = compute_patch_number_old(data.shape, sourceAxes, patchSize, patchStride)
    patchNumberFlat = np.prod(np.asarray(patchNumber), dtype=np.int64)  # type: int

    i = 0
    for patch, patchCenter, patchIndex in extract_patches_gen(data, sourceAxes, patchSize, patchStride,
                                                              verbose):
        if result is None:
            resultShape = (patchNumberFlat,) + patch.shape
            result = np.empty(resultShape, dtype=patch.dtype)
        result[i, ...] = patch
        patchCenters.append(patchCenter)
        i += 1

    return result, patchCenters, patchNumber


def extract_patches(data: Union[np.ndarray, h5py.Dataset], sourceAxes: Tuple, patchSize: Tuple,
                    patchStride: Tuple = None, batchSize=0, verbose=False):
    """
    Extract local patches/windows from the data.
    Source axes specify the dimensions that get cut into patches.
    The source axes get removed, a new axis enumerating patches is added as the first axis.

    E.g. split an 5-channel 2D image into a set of local 2D 5-channel patches;
    or split a dynamic volume into spatiotemporal windows.

    :param data:
    :param sourceAxes: Indices of the dimensions that get cut into slices.
    :param patchSize: Size of a single patch, including skipped axes.
    :param patchStride: Distance in data space between neighboring patches.
    :param batchSize:
    :param verbose: Whether to output progress information.
    :return:
    """

    return CppWrapper.extract_patches(data, sourceAxes, patchSize, patchStride, batchSize, verbose)


def extract_patches_batch(data: np.ndarray, sourceAxes: Tuple, patchSize: Tuple,
                          patchStride: Tuple = None, batchStart: int = 0, batchSize: int = 0):
    """
    The same as normal extract_patches, but extracts a single batch of patches.
    :param batchStart: First patch index in the batch.
    :param batchSize: The number of patches in the batch.
    :return:
    """

    return CppWrapper.extract_patches_batch(data, sourceAxes, patchSize, patchStride, batchStart, batchSize)


@deprecated("Use 'compute_patch_number' instead.")
def compute_patch_number_old(dataShape: Tuple, sourceAxes: Tuple, patchSize: Tuple,
                             patchStride: Tuple = None, patchInnerStride: Tuple = None,
                             lastFrameGap: int=1):
    """
    DEPRECATED
    This method is deprecated because it still uses 'source axes', which turned out to be not useful,
    and because it still pads the patch size by one to include the predicted frame.
    """
    if patchStride is None:
        patchStride = (1,) * len(dataShape)
    if patchInnerStride is None:
        patchInnerStride = (1,) * len(dataShape)

    patchNumber = []
    for i, axis in enumerate(sourceAxes):
        # How many voxels a patch covers.
        if i > 0:
            patchSupport = (patchSize[i] - 1) * patchInnerStride[i] + 1
        else:
            # Last point in time (Y-value) is 'lastFrameGap' frames away from the previous frame.
            # E.g. if 'lastFrameGap' is 1, it immediately follows it.
            patchSupport = (patchSize[i] - 2) * patchInnerStride[i] + 1 + lastFrameGap
        totalPatchNumber = dataShape[axis] - patchSupport + 1
        stride = patchStride[i]
        patchNumber.append(int(math.ceil(totalPatchNumber / stride)))

    return patchNumber


def compute_patch_number(volumeShape: Tuple, patchXSize: Tuple, patchYSize: Tuple,
                         patchStride: Tuple = None, patchInnerStride: Tuple = None,
                         predictionDelay: int = 1) -> Tuple:
    assert 4 <= len(volumeShape) <= 5
    volumeShape = volumeShape[:4]  # Ignore a potential feature dimension.

    if patchStride is None:
        patchStride = (1,) * len(volumeShape)
    if patchInnerStride is None:
        patchInnerStride = (1,) * len(volumeShape)

    patchSupport = compute_patch_support(patchXSize, patchYSize, patchInnerStride, predictionDelay)

    if any((patchSupport[dim] > volumeShape[dim] for dim in range(len(volumeShape)))):
        raise RuntimeError("Patch '{} == {} => {}' doesn't fit into the volume of size {}.".format(
            patchXSize, predictionDelay, patchYSize, volumeShape
        ))

    patchNumber = []
    for dim in range(len(volumeShape)):
        assert patchXSize[dim] >= patchYSize[dim]

        totalPatchNumber = volumeShape[dim] - patchSupport[dim] + 1
        stride = patchStride[dim]
        patchNumber.append(int(math.ceil(totalPatchNumber / stride)))

    return tuple(patchNumber)


def compute_patch_support(patchXSize: Tuple, patchYSize: Tuple,
                          patchInnerStride: Tuple = None, predictionDelay: int = 1) -> Tuple:
    """
    How many cells a single patch covers in each dimension (it's bounding box).
    Considers both the X- and the Y-patches.
    """

    patchSupport = [0] * len(patchXSize)
    for dim in range(len(patchXSize)):
        assert patchXSize[dim] >= patchYSize[dim]

        # How many voxels a patch covers.
        if dim > 0:
            patchSupport[dim] = (patchXSize[dim] - 1) * patchInnerStride[dim] + 1
        else:
            # Last point in time (Y-value) is 'predictionDelay' frames away from the previous frame.
            # E.g. if 'predictionDelay' is 1, it immediately follows it.
            patchSupport[dim] = (patchXSize[dim] - 1) * patchInnerStride[dim] + patchYSize[dim] + predictionDelay

    return tuple(patchSupport)


def compute_patch_y_offset(patchXSize: Tuple, patchYSize: Tuple,
                           patchInnerStride: Tuple, predictionDelay: int) -> Tuple:
    """
    Compute the offset of the Y-patch relative to the X-patch.
    """
    patchYOffset = [0] * len(patchXSize)
    for dim in range(len(patchXSize)):
        if dim == 0:
            patchYOffset[dim] = (patchXSize[dim] - 1) * patchInnerStride[dim] + predictionDelay
        else:
            patchYOffset[dim] = int(patchXSize[dim] / 2) * patchInnerStride[dim] - int(patchYSize[dim] / 2)

    return tuple(patchYOffset)


def compute_prediction_domain(volumeShape: Tuple,
                              patchXSize: Tuple,
                              patchYSize: Tuple,
                              patchStride,
                              patchInnerStride: Tuple,
                              predictionDelay: int) -> Tuple[Tuple, Tuple]:
    """
    Given patching parameters, computes the 'predictable' area in the data, i.e.
    the lowest and highest voxels predictable from valid input patches.

    :return:
    """
    assert 4 <= len(volumeShape) <= 5
    volumeShape = volumeShape[:4]  # Ignore a potential feature dimension.

    for dim in range(len(patchXSize)):
        assert patchXSize[dim] >= patchYSize[dim]

    patchNumber = compute_patch_number(volumeShape, patchXSize, patchYSize,
                                       patchStride, patchInnerStride, predictionDelay)
    patchYOffset = compute_patch_y_offset(patchXSize, patchYSize, patchInnerStride, predictionDelay)

    low = [0] * len(volumeShape)
    high = [0] * len(volumeShape)

    for dim in range(len(volumeShape)):
        lastPatchStart = (patchNumber[dim] - 1) * patchStride[dim]
        low[dim] = patchYOffset[dim]
        high[dim] = lastPatchStart + patchYOffset[dim] + patchYSize[dim]

    return (tuple(low), tuple(high))


@deprecated("Write a new function if this one is needed")
def patch_index_to_data_index_old(patchIndexFlat: int, dataShape: Tuple, sourceAxes: Tuple,
                                  patchSize: Tuple, patchStride: Tuple):
    """
    Convert a flat patch index into a data index which points to the lowest corner of the patch.
    Note: We don't need the inner stride or the last frame gap, because the lower corner
    doesn't depend on them.

    :param patchIndexFlat:
    :param dataShape:
    :param sourceAxes:
    :param patchSize:
    :param patchStride:
    :return:
    """

    ndim = len(dataShape)
    assert(sourceAxes == tuple(range(ndim)))  # Axis skipping is not implemented.

    patchNumber = compute_patch_number_old(dataShape, sourceAxes, patchSize, patchStride)
    patchIndexNd = npe.unflatten_index(patchNumber, patchIndexFlat)

    return tuple((patchIndexNd[dim] * patchStride[dim] for dim in range(ndim)))


def extract_patches_gen(data: Union[np.ndarray, h5py.Dataset], sourceAxes: Tuple[int], patchSize: Tuple,
                        patchStride: Tuple = None, verbose=False):
    """
    The same method but as a generator function.

    :param data:
    :param sourceAxes:
    :param patchSize:
    :param patchStride:
    :param verbose:
    :return:
    """
    # By default, extract every patch.
    if patchStride is None:
        patchStride = (1,) * len(sourceAxes)

    patchNumber = compute_patch_number_old(data.shape, sourceAxes, patchSize, patchStride)
    patchNumberFlat = np.prod(np.asarray(patchNumber), dtype=np.int64)

    lastPrintTime = time.time()

    # Since the number of traversed dimension is dynamically specified, we cannot use 'for' loops.
    # Instead, iterate a flat index and unflatten it inside the loop. (Easier than recursion.)
    for indexFlat in range(patchNumberFlat):
        patchIndexNd = npe.unflatten_index(patchNumber, indexFlat)
        # Since we skip some of the patches, scale the index accordingly.
        dataIndexNd = tuple(np.asarray(patchIndexNd, dtype=np.int) * np.asarray(patchStride, dtype=np.int))

        dataSelector = tuple()
        patchCenter = tuple()
        for axis in range(data.ndim):
            if axis not in sourceAxes:
                dataSelector += (slice(None),)  # Take everything, if it's not a source axis.
            else:
                patchAxis = sourceAxes.index(axis)
                # Take a slice along the axis.
                dataSelector += (slice(dataIndexNd[patchAxis], dataIndexNd[patchAxis] + patchSize[patchAxis]),)
                patchCenter += (dataIndexNd[patchAxis] + int(patchSize[patchAxis] / 2),)

        yield (data[dataSelector], patchCenter, patchIndexNd)

        if verbose and time.time() - lastPrintTime > 20:
            lastPrintTime = time.time()
            print("Extracting patches: {:02.2f}%".format(indexFlat / patchNumberFlat * 100))


def extract_patched_training_data_without_empty_4d(data: np.ndarray, dataShape: Tuple, dataStartFlat: int,
                                                   dataEndFlat: int, outputX: np.ndarray, outputY: np.ndarray,
                                                   outputIndices: np.ndarray, patchSize: Tuple, patchStride: Tuple,
                                                   batchStartIndex: int, batchSize: int,
                                                   patchInnerStride: Tuple = (1, 1, 1, 1),
                                                   lastFrameGap: int = 1,
                                                   undersamplingProb: float = 1.0,
                                                   skipEmptyPatches: bool = True, emptyValue: int = 0):
    """
    Extract patches/windows from a 4-dimensional array.
    Each patch gets split into training data: X and Y.
    X holds the whole hypercube, except for the last frame. Y holds a single scalar
    from the center of the last frame. (Time is the first dimension, C-order is assumed.)
    'Empty' patches are those, where all values in X and the Y value are equal to the 'empty value'.
    Empty patches do not get extracted.
    Extraction is performed in batches, returning control after 'batchSize' patches were extracted.
    """
    return CppWrapper.extract_patched_training_data_without_empty_4d(data, dataShape, dataStartFlat, dataEndFlat,
                                                                     outputX, outputY, outputIndices, patchSize,
                                                                     patchStride, batchStartIndex, batchSize,
                                                                     patchInnerStride=patchInnerStride,
                                                                     lastFrameGap=lastFrameGap,
                                                                     undersamplingProb=undersamplingProb,
                                                                     skipEmptyPatches=skipEmptyPatches,
                                                                     emptyValue=emptyValue)


def extract_patched_training_data_without_empty_4d_multi(data: np.ndarray, dataShape: Tuple, dataStartFlat: int,
                                                         dataEndFlat: int, outputX: np.ndarray, outputY: np.ndarray,
                                                         outputIndices: np.ndarray, patchSize: Tuple,
                                                         patchStride: Tuple,
                                                         batchStartIndex: int, batchSize: int,
                                                         patchInnerStride: Tuple = (1, 1, 1, 1),
                                                         lastFrameGap: int = 1,
                                                         undersamplingProb: float = 1.0,
                                                         skipEmptyPatches: bool = True, emptyValue: int = 0,
                                                         threadNumber: int = 8):
    """
    Same as 'extract_patched_training_data_without_empty_4d', but multithreaded.
    Because of this, the output patches aren't in any particular order,
    though all the patches are guaranteed to be extracted.
    """

    return CppWrapper.extract_patched_training_data_without_empty_4d_multi(data, dataShape, dataStartFlat, dataEndFlat,
                                                                           outputX, outputY, outputIndices,
                                                                           patchSize, patchStride,
                                                                           batchStartIndex, batchSize,
                                                                           patchInnerStride=patchInnerStride,
                                                                           lastFrameGap=lastFrameGap,
                                                                           undersamplingProb=undersamplingProb,
                                                                           skipEmptyPatches=skipEmptyPatches,
                                                                           emptyValue=emptyValue,
                                                                           threadNumber=threadNumber)


def extract_patched_training_data_4d_multithreaded(data: np.ndarray, dataShape: Tuple, dataStartFlat: int,
                                                   dataEndFlat: int, outputX: np.ndarray, outputY: np.ndarray,
                                                   patchSize: Tuple, batchStartIndex: int, batchSize: int,
                                                   patchInnerStride: Tuple = (1, 1, 1, 1), lastFrameGap: int = 1,
                                                   threadNumber: int = 8):
    """
    Same as 'extract_patched_training_data_without_empty_4d', but without empty patch skipping
    which allows multiple threads to be used for extraction, while maintaining correct output order.
    """
    return CppWrapper.extract_patched_training_data_4d_multithreaded(data, dataShape, dataStartFlat, dataEndFlat, outputX,
                                                                     outputY, patchSize,
                                                                     batchStartIndex, batchSize,
                                                                     patchInnerStride=patchInnerStride,
                                                                     lastFrameGap=lastFrameGap,
                                                                     threadNumber=threadNumber)


def extract_patched_all_data_without_empty_4d(data: np.ndarray,
                                              patchSize: Tuple, patchStride: Tuple, emptyValue: int,
                                              batchSize=10000):
    """
    A wrapper around 'extract_patched_training_data_without_empty_4d', takes care of managing
    batching, extracting all available data in one call.
    Also serves as a documentation for how to use the single-batch method.

    :param data:
    :param patchSize:
    :param patchStride:
    :param emptyValue:
    :param batchSize:
    :return:
    """
    dataSizeFlat = npe.multiply(data.shape) * data.dtype.itemsize
    batchSize = min(batchSize, data.size)

    patchSizeX = (patchSize[0] - 1,) + patchSize[1:]

    allDataX = npe.NumpyDynArray((-1,) + patchSizeX, dtype=data.dtype)
    allDataY = npe.NumpyDynArray((-1, 1), dtype=data.dtype)
    allDataIndices = npe.NumpyDynArray((-1, data.ndim), dtype=np.uint64)

    batchX = np.empty((batchSize,) + patchSizeX, dtype=data.dtype)
    batchY = np.empty((batchSize, 1), dtype=data.dtype)
    batchIndices = np.empty((batchSize, data.ndim), dtype=np.uint64)

    patchesExtracted = batchSize
    nextBatchIndex = 0
    while patchesExtracted == batchSize:
        patchesExtracted, nextBatchIndex, dataEndReached = \
            extract_patched_training_data_without_empty_4d(data, data.shape, 0, dataSizeFlat, batchX, batchY,
                                                           batchIndices, patchSize, patchStride, nextBatchIndex,
                                                           batchSize, skipEmptyPatches=True, emptyValue=emptyValue)

        allDataX.append_all(batchX[:patchesExtracted])
        allDataY.append_all(batchY[:patchesExtracted])
        allDataIndices.append_all(batchIndices[:patchesExtracted])

    return allDataX.get_all(), allDataY.get_all(), allDataIndices.get_all()


def get_patch_from_volume(volumeData: npe.LargeArray,
                          patchCoords: Tuple[int, ...],
                          patchSize: Tuple[int, ...]) -> np.ndarray:
    selector = tuple(slice(patchCoords[dim], patchCoords[dim] + patchSize[dim])
                     for dim in range(len(patchSize)))

    patch = volumeData[selector]
    # We check only the first N dimensions, in case there's an attribute dimension.
    if any(patch.shape[d] != patchSize[d] for d in range(len(patchSize))):
        raise ValueError("Invalid patch size or coords. Patch of size {} at pos {} doesn't fit into volume of shape {}."
                         .format(patchSize, patchCoords, volumeData.shape))

    return patch


