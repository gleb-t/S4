import random
import json
import time
import logging
from typing import Tuple, Iterable, Union, Type

import numpy as np
import scipy.signal
import h5py

import PythonExtras.logging_tools as logging_tools
from PythonExtras.CppWrapper import CppWrapper
from PythonExtras.BufferedNdArray import BufferedNdArray

LargeArray = Union[np.ndarray, h5py.Dataset, BufferedNdArray]


def multiply(iterable: Iterable):
    prod = 1
    for x in iterable:
        prod *= x
    return prod


def slice_len(sliceObject: slice, sequenceLength: int):
    return len(range(*sliceObject.indices(sequenceLength)))


def slice_shape(shape: Tuple[int, ...], slices: Tuple[slice, ...]) -> Tuple[int, ...]:
    assert len(slices) == len(shape)

    result = [0] * len(slices)
    for dim in range(len(slices)):
        s = slices[dim]  # type: slice
        assert (all((x or 0) >= 0 for x in [s.start, s.stop, s.step]))  # All positive.

        startIndex = s.start or 0
        endIndex = min(s.stop or shape[dim], shape[dim])
        length = max(endIndex - startIndex, 0)
        step = s.step or 1

        result[dim] = 1 + int((length - 1) / step)

    return tuple(result)


def slice_nd(start, end):
    """
    Take a slice of an nd-array along all axes simultaneously..
    :param start:
    :param end:
    :return:
    """
    assert len(start) == len(end)

    result = []
    for i in range(0, len(start)):
        result.append(slice(start[i], end[i]))

    return tuple(result)


def slice_along_axis(index, axis, ndim):
    """
    Return a selector that takes a single-element slice (subtensor) of an nd-array
    along a certain axis (at a given index).
    The result would be an (n-1)-dimensional array. E.g. data[:, :, 5, :].
    The advantage of this function over subscript syntax is that you can
    specify the axis with a variable.

    :param index:
    :param axis:
    :param ndim:
    :return:
    """
    return tuple([index if axis == i else None for i in range(0, ndim)])


def compute_byte_size(data: LargeArray):
    elementNumber = multiply(data.shape)
    elementSize = data.dtype.itemsize

    return elementNumber * elementSize


def weighted_mean(data, weights, axis=None):

    assert axis == 0
    assert weights.ndim == 1

    weightSum = np.sum(weights)
    weightsNorm = weights / weightSum
    return np.sum(data * weightsNorm[:, np.newaxis], axis=0)


def weighted_variance(data, weights, axis=None, weightedMean=None):
    assert axis == 0
    assert weights.ndim == 1

    if weightedMean is None:
        weightedMean = weighted_mean(data, weights, axis=axis)

    weightSum = np.sum(weights)
    weightsNorm = weights / weightSum

    squaredDeviation = (data - weightedMean) ** 2
    return np.sum(squaredDeviation * weightsNorm[:, np.newaxis], axis=0)


def weighted_std(data, weights, axis=None, weightedMean=None):
    return np.sqrt(weighted_variance(data, weights, axis, weightedMean))


def moving_average_nd(data, kernelSize):
    """
    Perform local averaging of an Nd-array.
    Since averaging kernel is separable, convolution is applied iteratively along each axis..

    :param data:
    :param kernelSize:
    :return:
    """

    kernels = []
    for dim in range(data.ndim):
        kernelShape = tuple((min(kernelSize, data.shape[dim]) if dim == i else 1 for i in range(data.ndim)))
        kernels.append(np.ones(kernelShape) / kernelSize)

    result = data.copy()  # Doesn't work correctly in-place.
    for kernel in kernels:
        result = scipy.signal.convolve(result, kernel, 'same')

    return result


def argsort2d(data):
    """
    Return an array of indices into a sorted version of a 2d-array.
    :param data:
    :return:
    """
    assert(data.ndim == 2)
    origShape = data.shape
    dataFlat = data.reshape(-1)
    indicesSorted = np.argsort(dataFlat)
    indicesMulti = [np.unravel_index(i, origShape) for i in indicesSorted.tolist()]

    return indicesMulti


def unflatten_index(shape, indexFlat):
    """
    Convert a flat index into an N-d index (a tuple).
    :param shape:
    :param indexFlat:
    :return:
    """
    indexNd = tuple()
    for axis, length in enumerate(shape):
        sliceSize = np.prod(shape[axis + 1:], dtype=np.int64)
        axisIndex = int(indexFlat / sliceSize)
        indexNd += (axisIndex,)
        indexFlat -= axisIndex * sliceSize

    return indexNd


def flatten_index(indexTuple, size):
    """
    Converts a Nd index into a flat array index. C order is assumed.
    (The point is that it works with any number of dimensions.)
    :param indexTuple:
    :param size:
    :return:
    """
    ndim = len(size)
    assert(ndim == len(indexTuple))

    sliceSizes = np.empty(ndim, type(size[0]))
    sliceSizes[-1] = 1
    for i in range(ndim - 2, -1, -1):
        sliceSizes[i] = size[i + 1] * sliceSizes[i + 1]

    flatIndex = 0
    for i in range(0, ndim):
        flatIndex += indexTuple[i] * sliceSizes[i]

    return flatIndex


def tuple_corners_to_slices(low: Tuple, high: Tuple) -> Tuple:
    """
    Convert two tuples representing low and high corners of an nd-box
    into a tuple of slices that can be used to select that box out of a volume.

    :param low:
    :param high:
    :return:
    """

    return tuple(slice(low[dim], high[dim]) for dim in range(len(low)))


def get_batch_indices(shape: Tuple, dtype: Type=None, batchSizeFlat=1e8, batchSize=None):
    """
    Batching helper. Returns a pair (start and end) of indices for each batch.
    Batches along axis 0.

    :param shape:
    :param dtype:
    :param batchSizeFlat:
    :param batchSize:
    :return:
    """
    if batchSize is None:
        if not isinstance(dtype, np.dtype):
            # noinspection PyCallingNonCallable
            dtype = dtype()  # Instantiate the dtype class.

        sliceSizeFlat = np.prod(np.asarray(shape[1:]), dtype=np.int64) * dtype.itemsize
        batchSize = int(batchSizeFlat // sliceSizeFlat)
        batchSize = min(batchSize, shape[0])

        if batchSize == 0:
            raise RuntimeError("Batch size '{}' is too small to fit a volume slice of size '{}'"
                               .format(batchSizeFlat, sliceSizeFlat))

    for batchStart in range(0, shape[0], batchSize):
        batchEnd = batchStart + min(batchSize, shape[0] - batchStart)
        yield (batchStart, batchEnd)


def shuffle_hdf_arrays_together(dataA: h5py.Dataset, dataB: h5py.Dataset, blockSize: int=1, logger: logging.Logger=None):
    """
    Shuffle rows of two HDF array in sync: corresponding rows end up at the same place.
    Used for shuffling X and Y training data arrays.
    :param dataA:
    :param dataB:
    :param blockSize: Specifies the number of rows that will be moved together as a single element.
                      Rows won't be shuffled inside of a block.
                      Set to one for true shuffling, use a larger value to perform an approximate shuffle.
    :param logger:
    :return:
    """
    # Explicitly make sure we have an HDF dataset, not a numpy array, since we make some assumptions based on it.
    if not hasattr(dataA, 'id') or not hasattr(dataB, 'id'):  # Don't check the type, it's some weird internal class.
        raise ValueError('Provided input is not in HDF datasets.')

    if dataA.shape[0] != dataB.shape[0]:
        raise ValueError('Arguments should have equal length, {} and {} given.'.format(dataA.shape[0], dataB.shape[0]))

    timeStart, timeLastReport = time.time(), time.time()

    length = dataA.shape[0]
    blockNumber = length // blockSize
    for iSource in range(0, blockNumber - 2):  # First to next-to-last. (Simplifies the block size check.)
        iTarget = int(random.uniform(iSource, blockNumber - 1e-10))  # [i, blockNumber)
        # The last block might need to be shortened, if array length is now divisible by the block size.
        actualBlockSize = blockSize if iTarget < blockNumber - 1 else length - iTarget * blockSize

        sourceSelector = slice(iSource * blockSize, iSource * blockSize + actualBlockSize)
        targetSelector = slice(iTarget * blockSize, iTarget * blockSize + actualBlockSize)

        # Swap data in array A.
        temp = dataA[targetSelector]  # This creates a copy, since we read from HDF, no need to do it explicitly.
        dataA[targetSelector] = dataA[sourceSelector]
        dataA[sourceSelector] = temp

        # Swap data synchronously in array B.
        temp = dataB[targetSelector]
        dataB[targetSelector] = dataB[sourceSelector]
        dataB[sourceSelector] = temp

        # Report current progress, but not too often.
        if logger is not None and iSource % 100 == 0:
            logger.info("Shuffling an HDF array ...{:.2f}% in {}"
                        .format(iSource / blockNumber * 100,
                                logging_tools.format_duration(time.time() - timeStart)),
                        extra={'throttlingId': 'shuffle_hdf_arrays_together'})


def abs_diff_hdf_arrays(dataA: h5py.Dataset, dataB: h5py.Dataset, output: h5py.Dataset,
                        dtype: np.dtype, batchSizeFlat=1e8):
    """
    Compute element-wise |A-B|.
    Computation is performed in batches to decrease memory requirements.

    :param dtype:
    :param batchSizeFlat:
    :param dataA:
    :param dataB:
    :param output:
    :return:
    """
    if dataA.shape != dataB.shape or dataA.shape != output.shape:
        raise ValueError("Arguments should have equal shapes, {}, {} and {} given."
                         .format(dataA.shape, dataB.shape, output.shape))
    if output.dtype != dtype:
        raise ValueError("Output array data type doesn't match the provided type: {} and {}"
                         .format(output.dtype, dtype))

    for batchStart, batchEnd in get_batch_indices(dataA.shape, dtype, batchSizeFlat):
        output[batchStart:batchEnd] = np.abs(dataA[batchStart:batchEnd].astype(dtype) -
                                             dataB[batchStart:batchEnd].astype(dtype))


def abs_diff_hdf_arrays_masked(dataA: h5py.Dataset, dataB: h5py.Dataset, mask: h5py.Dataset, output: h5py.Dataset,
                               dtype: Type, batchSizeFlat=1e8):
    """
    Compute element-wise |A-B| where mask is set to true, zero everywhere else.
    Computation is performed in batches to decrease memory requirements.

    :param dataA:
    :param dataB:
    :param mask:
    :param output:
    :param dtype:
    :param batchSizeFlat:
    :return:
    """
    if dataA.shape != dataB.shape or dataA.shape != output.shape or dataA.shape != mask.shape:
        raise ValueError("Arguments should have equal shapes, {}, {}, {} and {} given."
                         .format(dataA.shape, dataB.shape, mask.shape, output.shape))
    if output.dtype != dtype:
        raise ValueError("Output array data type doesn't match the provided type: {} and {}"
                         .format(output.dtype, dtype))

    for batchStart, batchEnd in get_batch_indices(dataA.shape, dtype, batchSizeFlat):
        batchMask = mask[batchStart:batchEnd]
        batchDiff = batchMask * np.abs(dataA[batchStart:batchEnd].astype(dtype) -
                                       dataB[batchStart:batchEnd].astype(dtype))
        output[batchStart:batchEnd] = batchDiff


def mse_large_arrays(dataA: 'LargeArray', dataB: 'LargeArray', dtype: Type, batchSizeFlat=1e8):
    """
    Compute MSE between two HDF datasets.
    Computation is performed in batches to decrease memory requirements.

    """
    if dataA.shape != dataB.shape:
        raise ValueError("Arguments should have equal shapes, {} and {} given."
                         .format(dataA.shape, dataB.shape))

    sum = 0.0
    count = 0
    for batchStart, batchEnd in get_batch_indices(dataA.shape, dtype, batchSizeFlat):
        batchA = dataA[batchStart:batchEnd].astype(dtype)
        batchB = dataB[batchStart:batchEnd].astype(dtype)
        sum += np.sum(np.square(batchA - batchB), dtype=dtype)
        count += multiply(batchA.shape)

    return sum / count if count > 0 else float('nan')


def mse_large_arrays_masked(dataA: 'LargeArray', dataB: 'LargeArray', mask: 'LargeArray',
                            dtype: Type, batchSizeFlat=1e8):
    """
    Compute MSE between two HDF datasets, considering elements where the mask is set to true (one).
    Computation is performed in batches to decrease memory requirements.
    """
    if dataA.shape != dataB.shape or dataA.shape != mask.shape:
        raise ValueError("Arguments should have equal shapes, {}, {} and {} given."
                         .format(dataA.shape, dataB.shape, mask.shape))

    sum = 0.0
    count = 0
    for batchStart, batchEnd in get_batch_indices(dataA.shape, dtype, batchSizeFlat):
        batchMask = mask[batchStart:batchEnd]
        diff = batchMask * (dataA[batchStart:batchEnd].astype(dtype) - dataB[batchStart:batchEnd].astype(dtype))
        square = np.square(diff)
        nonzeroNumber = np.count_nonzero(batchMask)
        sum += np.sum(square)
        count += nonzeroNumber

    return sum / count if count > 0 else float('nan')


def var_large_array_masked(data: 'LargeArray', mask: 'LargeArray', dtype: Type, batchSizeFlat=1e8):
    """
    Compute MSE between two HDF datasets, considering elements where the mask is set to true.
    Computation is performed in batches to decrease memory requirements.
    """

    sum = 0.0
    count = 0
    for batchStart, batchEnd in get_batch_indices(data.shape, dtype, batchSizeFlat):
        batchMask = mask[batchStart:batchEnd].astype(dtype)
        batchSum = np.sum((batchMask * data[batchStart:batchEnd].astype(dtype)), dtype=dtype)
        nonzeroNumber = np.count_nonzero(batchMask)
        sum += batchSum
        count += nonzeroNumber
    mean = sum / count if count > 0 else float('nan')

    sum = 0.0
    count = 0
    for batchStart, batchEnd in get_batch_indices(data.shape, dtype, batchSizeFlat):
        batchMask = mask[batchStart:batchEnd].astype(dtype)
        batchSum = np.sum(batchMask * np.square(data[batchStart:batchEnd].astype(dtype) - mean), dtype=dtype)
        nonzeroNumber = np.count_nonzero(batchMask)
        sum += batchSum
        count += nonzeroNumber

    return sum / count if count > 0 else float('nan')


def var_large_array(data: 'LargeArray', dtype: Type, batchSizeFlat=1e8):
    """
    Compute MSE between two HDF datasets.
    Computation is performed in batches to decrease memory requirements.

    :param dtype:
    :param batchSizeFlat:
    :param data:
    :param dataB:
    :param output:
    :return:
    """

    sum = 0.0
    count = 0
    for batchStart, batchEnd in get_batch_indices(data.shape, dtype, batchSizeFlat):
        batch = data[batchStart:batchEnd]
        sum += np.sum(batch.astype(dtype), dtype=dtype)
        count += multiply(batch.shape)
    mean = sum / count if count > 0 else float('nan')

    sum = 0.0
    count = 0
    for batchStart, batchEnd in get_batch_indices(data.shape, dtype, batchSizeFlat):
        batch = data[batchStart:batchEnd]
        sum += np.sum(np.square(batch.astype(dtype) - mean), dtype=dtype)
        count += multiply(batch.shape)

    return sum / count if count > 0 else float('nan')


def reshape_and_pad_volume(inputFlatVolume: h5py.Dataset, outputVolume: h5py.Dataset, targetDomain: Tuple[Tuple, Tuple]):
    """
    Reshapes a flat input array into a 4D volume and 'pastes' it into the output,
    such that it occupies the target domain.

    See also: get_prediction_domain()

    :param inputFlatVolume:
    :param outputVolume:
    :param targetDomain: min and max corners
    :return:
    """

    assert(inputFlatVolume.ndim == 2)
    assert(outputVolume.ndim == 4)

    targetBegin = np.asarray(targetDomain[0])
    targetEnd = np.asarray(targetDomain[1])
    targetShape = targetEnd - targetBegin
    targetSpatialSizeFlat = int(np.prod(targetShape[1:]))  # Cast to python int to avoid overflow.

    # Do frame-by-frame to keep things out-of-core.
    for fInput in range(targetShape[0]):

        outputSpatialSelector = tuple(slice(targetBegin[dim], targetEnd[dim])
                                      for dim in range(1, 4))
        inputFlatSelector = slice(fInput * targetSpatialSizeFlat,
                                  (fInput + 1) * targetSpatialSizeFlat)
        fOutput = targetBegin[0] + fInput

        outputVolume[(fOutput,) + outputSpatialSelector] = inputFlatVolume[inputFlatSelector, :].reshape(targetShape[1:])


def sparse_insert_into_bna(bna: 'BufferedNdArray', indices: np.ndarray, values: np.ndarray, valueNumber: int):
    """
    Insert values at given indices into the nd-array.
    Much more efficient than looping in Python.
    """
    return CppWrapper.sparse_insert_into_bna(bna, indices, values, valueNumber)


def sparse_insert_slices_into_bna(bna: 'BufferedNdArray', indices: np.ndarray, slices: np.ndarray,
                                  sliceNdim: int, sliceNumber: int):
    """
    Insert slices at given indices into the nd-array. Slices have to be along the last 'sliceDim' axes.
    Much more efficient than looping in Python.
    """
    return CppWrapper.sparse_insert_slices_into_bna(bna, indices, slices, sliceNdim, sliceNumber)


def sparse_insert_patches_into_bna(bna: 'BufferedNdArray', indices: np.ndarray, patches: np.ndarray,
                                   patchSize: Tuple, patchNumber: int, isConstPatch: bool):
    """
    Insert patches at given indices into the nd-array. Patches have to be along all the dimensions.
    If 'isConstPatch' is true, expect 'patches' to contain only a single patch.
    Much more efficient than looping in Python.
    """
    return CppWrapper.sparse_insert_patches_into_bna(bna, indices, patches, patchSize, patchNumber, isConstPatch)


def sparse_insert_const_into_bna(bna: 'BufferedNdArray', indices: np.ndarray, value: int, valueNumber: int):
    """
    Insert a constant values at given indices into the nd-array.
    Much more efficient than looping in Python.
    """
    return CppWrapper.sparse_insert_const_into_bna(bna, indices, value, valueNumber)


def smooth_3d_array_average(data: np.ndarray, kernelRadius: int) -> np.ndarray:
    return CppWrapper.smooth_3d_array_average(data, kernelRadius)


def aggregate_attention_volume(attentionVolumeFull: 'BufferedNdArray', dataSize: Tuple, patchXSize: Tuple,
                               predictionStride: Tuple, attentionAgg: 'BufferedNdArray'):
    """
    Aggregate an 8D attention volume into 4D. Each output voxel represents the sum over all
    attention vectors covering that value. I.e. it's voxel's total 'importance'.
    """
    return CppWrapper.aggregate_attention_volume(attentionVolumeFull, dataSize, patchXSize,
                                                 predictionStride, attentionAgg)


def aggregate_attention_volume_local_attention(attentionVolumeFull: 'BufferedNdArray',
                                               outAttentionAvg: np.ndarray, outAttentionVar: np.ndarray):
    """
    Aggregate an 8D attention volume into a single 4D attention patch.
    Calculate both the mean attention value and the STD.
    """
    return CppWrapper.aggregate_attention_volume_local_attention(attentionVolumeFull, outAttentionAvg, outAttentionVar)


class JsonEncoder(json.JSONEncoder):
    """
    A custom JSON encoder class that is required for encoding Numpy array values
    without explicitly converting them to Python types.
    Without it, JSON serializer will crash, since Numpy types aren't 'serializable'.
    Credit: https://stackoverflow.com/a/27050186/1545327
    """

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, slice):
            return '__slice({}, {}, {})'.format(o.start, o.stop, o.step)
        if isinstance(o, object):
            return {'__type': type(o).__qualname__, **o.__dict__}

        return super().default(o)


class NumpyDynArray:
    """
    A dynamically resized ND-array that uses NumPy for storage.
    """
    def __init__(self, shape, dtype=None):
        # For now we assume that the first dimension is the variable-sized dimension.
        assert(shape[0] == -1)

        self.size = 0
        self.capacity = 100
        shape = (self.capacity,) + shape[1:]

        self.data = np.empty(shape, dtype=dtype)

    def __str__(self, *args, **kwargs):
        return self.get_all().__str__(*args, **kwargs)

    @property
    def shape(self):
        return self.get_all().shape  # Should be fast, assuming that get_all returns a view.

    def append(self, dataToAdd):
        """
        Append a single row (subtensor?) of data along the dynamic axis.
        :param dataToAdd:
        :return:
        """
        if dataToAdd.shape != self.data.shape[1:]:
            raise ValueError("Cannot append the data. Expected shape: {}. Provided: {}."
                             .format(self.data.shape[1:], dataToAdd.shape))

        # Do we still have free space?
        if self.size >= self.capacity:
            self._allocate_more_space()

        # Add a new 'row' (hunk).
        self.data[self.size, ...] = dataToAdd
        self.size += 1

    def append_all(self, dataToAdd):
        """
        Append multiple rows (subtensors?) of data. Effectively a concatenation
        along the dynamic axis.
        :param dataToAdd:
        :return:
        """
        if dataToAdd.shape[1:] != self.data.shape[1:]:
            raise ValueError("Cannot append the data. Expected shape: {}. Provided: {}."
                             .format(self.data.shape[1:], dataToAdd.shape))

        newRowNumber = dataToAdd.shape[0]
        # Do we still have free space?
        while self.size + newRowNumber >= self.capacity:
            self._allocate_more_space()

        self.data[self.size:self.size + newRowNumber, ...] = dataToAdd
        self.size += newRowNumber

    def _allocate_more_space(self):
        # Allocate a new array of a bigger size.
        self.capacity *= 2
        shape = (self.capacity,) + self.data.shape[1:]
        newData = np.empty(shape, dtype=self.data.dtype)
        # Copy the data to the new array.
        newData[:self.size, ...] = self.data[:self.size, ...]
        self.data = newData

    def get_all(self):
        return self.data[:self.size, ...]

    def __getitem__(self, item):
        return self.data.__getitem__(item)

    def __setitem__(self, key, value):
        raise NotImplementedError()
