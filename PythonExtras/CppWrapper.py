import ctypes
from typing import Tuple, Union, List
import h5py
import os
import platform
import time
import math
import warnings
from typing import Callable, Tuple, List, Dict, Any, Union, Type, TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from PythonExtras.BufferedNdArray import BufferedNdArray
    from PythonExtras.MtPatchExtractor import MtPatchExtractor


# todo Rework this class to just help calling the DLL code, instead of wrapping every function.
# Let the client have the wrapper.
class CppWrapper:
    """
    Wraps functions written in C++ into Python code.
    """

    _Dll = None

    @classmethod
    def _get_c_dll(cls):
        if cls._Dll is None:
            platformName = platform.system()
            if platformName == 'Windows':
                dllPath = os.path.join(os.path.dirname(__file__), 'c_dll', 'PythonExtrasC.dll')
                cls._Dll = ctypes.CDLL(dllPath)
            elif platformName == 'Linux':
                soPath = os.path.join(os.path.dirname(__file__), 'c_dll', 'PythonExtrasC.so')
                cls._Dll = ctypes.CDLL(soPath)
            else:
                raise RuntimeError("Unknown platform: {}".format(platformName))

        return cls._Dll

    @classmethod
    def test(cls, data: np.ndarray):
        func = cls._get_c_dll().test
        func(ctypes.c_void_p(data.ctypes.data), ctypes.c_int32(4), ctypes.c_int32(4))

    @classmethod
    def static_state_test(cls, increment: int):
        func = cls._get_c_dll().static_state_test
        result = ctypes.c_uint64()
        func(ctypes.c_uint64(increment), ctypes.byref(result))

        return result.value

    @classmethod
    def multithreading_test(cls, data: np.ndarray, threadNumber: int):
        func = cls._get_c_dll().multithreading_test
        size = data.shape[0]

        func(data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), ctypes.c_uint64(size), ctypes.c_uint64(threadNumber))

        return data

    @classmethod
    def resize_array_point(cls, data: np.ndarray, targetSize: Tuple):
        assert(data.ndim <= 3)

        # Implicitly support 1D and 3D array by adding extra dimensions.
        ndimOrig = data.ndim
        dataShapeOrig = data.shape
        while data.ndim < 3:
            data = data[..., np.newaxis]
            targetSize += (1,)

        if data.dtype == np.float32:
            func = cls._get_c_dll().resize_array_point_float32
        elif data.dtype == np.float64:
            func = cls._get_c_dll().resize_array_point_float64
        elif data.dtype == np.uint8:
            func = cls._get_c_dll().resize_array_point_uint8
        else:
            raise(RuntimeError("Unsupported data type: {}".format(data.dtype)))

        output = np.empty(targetSize, dtype=data.dtype)

        func(ctypes.c_void_p(data.ctypes.data),
             ctypes.c_int32(data.shape[0]),
             ctypes.c_int32(data.shape[1]),
             ctypes.c_int32(data.shape[2]),
             ctypes.c_void_p(output.ctypes.data),
             ctypes.c_int32(targetSize[0]),
             ctypes.c_int32(targetSize[1]),
             ctypes.c_int32(targetSize[2]))

        while output.ndim > ndimOrig:
            output = output[..., 0]

        return output

    @classmethod
    def extract_patches(cls, data: Union[np.ndarray, h5py.Dataset], sourceAxes: Tuple, patchSize: Tuple,
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
            :param batchSize: How many patches to process per C++ call: non-zero values allow for
                              progress reporting.
            :param verbose: Whether to output progress information.
            :return:
            """

        assert(data.dtype == np.uint8)

        # By default, extract every patch.
        if patchStride is None:
            patchStride = (1,) * len(sourceAxes)

        patchShape = cls._compute_patch_shape(data.shape, sourceAxes, patchSize)
        patchNumber = cls._compute_patch_number(data.shape, sourceAxes, patchSize, patchStride)
        patchNumberFlat = np.prod(np.asarray(patchNumber))  # type: int

        inputData = data[...]  # type: np.ndarray
        outputBufferSize = patchNumberFlat * np.prod(np.asarray(patchShape)) * data.dtype.itemsize
        outputData = np.empty((patchNumberFlat,) + tuple(patchShape), dtype=data.dtype)
        outputCenters = np.empty((patchNumberFlat, len(sourceAxes)), dtype=np.uint64)

        func = cls._get_c_dll().extract_patches_batched_uint8

        dataSize = np.array(data.shape, dtype=np.uint64)
        sourceAxes = np.array(sourceAxes, dtype=np.uint64)
        patchSize = np.array(patchSize, dtype=np.uint64)
        patchStride = np.array(patchStride, dtype=np.uint64)

        if batchSize == 0:
            batchSize = patchNumberFlat

        for i in range(0, patchNumberFlat, batchSize):
            patchesInBatch = min(batchSize, patchNumberFlat - i)

            timeStart = time.time()
            func(ctypes.c_void_p(inputData.ctypes.data),
                 ctypes.c_void_p(outputData.ctypes.data),
                 outputCenters.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                 ctypes.c_uint64(inputData.ndim),
                 dataSize.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                 ctypes.c_uint64(dataSize.size),
                 sourceAxes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                 ctypes.c_uint64(sourceAxes.size),
                 patchSize.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                 ctypes.c_uint64(patchSize.size),
                 patchStride.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                 ctypes.c_uint64(patchStride.size),
                 ctypes.c_uint64(i), ctypes.c_uint64(patchesInBatch), ctypes.c_bool(False))

            if verbose:
                print("Extracting patches: {:02.2f}%, {:02.2f}s. per batch".format(i / patchNumberFlat * 100,
                                                                                   time.time() - timeStart))

        return outputData, outputCenters, patchNumber

    @classmethod
    def extract_patches_batch(cls, data: np.ndarray, sourceAxes: Tuple, patchSize: Tuple,
                              patchStride: Tuple = None, batchStart: int = 0, batchSize: int = 0):
        assert (data.dtype == np.uint8)
        # Because this batched function is called multiple times, we do not manage h5py datasets inside.
        assert (isinstance(data, np.ndarray))

        # By default, extract every patch.
        if patchStride is None:
            patchStride = (1,) * len(sourceAxes)

        patchShape = cls._compute_patch_shape(data.shape, sourceAxes, patchSize)
        patchNumber = cls._compute_patch_number(data.shape, sourceAxes, patchSize, patchStride)
        patchNumberFlat = np.prod(np.asarray(patchNumber))  # type: int

        if batchSize == 0:
            batchSize = patchNumberFlat

        patchesInBatch = min(batchSize, patchNumberFlat - batchStart)

        inputData = data
        outputData = np.empty((patchesInBatch,) + tuple(patchShape), dtype=data.dtype)
        outputCenters = np.empty((patchesInBatch, len(sourceAxes)), dtype=np.uint64)

        func = cls._get_c_dll().extract_patches_batched_uint8

        dataSize = np.array(data.shape, dtype=np.uint64)
        sourceAxes = np.array(sourceAxes, dtype=np.uint64)
        patchSize = np.array(patchSize, dtype=np.uint64)
        patchStride = np.array(patchStride, dtype=np.uint64)

        func(ctypes.c_void_p(inputData.ctypes.data),
             ctypes.c_void_p(outputData.ctypes.data),
             outputCenters.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             ctypes.c_uint64(inputData.ndim),
             dataSize.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             ctypes.c_uint64(dataSize.size),
             sourceAxes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             ctypes.c_uint64(sourceAxes.size),
             patchSize.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             ctypes.c_uint64(patchSize.size),
             patchStride.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             ctypes.c_uint64(patchStride.size),
             ctypes.c_uint64(batchStart), ctypes.c_uint64(patchesInBatch), ctypes.c_bool(True))

        return outputData, outputCenters, patchNumber

    @classmethod
    def extract_patched_training_data_without_empty_4d(cls, dataBuffer: np.ndarray, dataShape: Tuple,
                                                       dataStartFlat: int, dataEndFlat: int, outputX: np.ndarray,
                                                       outputY: np.ndarray, outputIndices: np.ndarray, patchSize: Tuple,
                                                       patchStride: Tuple, batchStartIndex: int, batchSize: int,
                                                       patchInnerStride: Tuple = (1, 1, 1, 1), lastFrameGap:int = 1,
                                                       undersamplingProb: float = 1.0,
                                                       skipEmptyPatches: bool = True, emptyValue: int = 0):
        assert(len(dataShape) == 4)
        # C++ code copies data column-by-column along the last axis,
        # this means that the last axis must be continuous.
        assert(patchInnerStride[-1] == 1)
        assert(dataBuffer.dtype == np.uint8)
        # Because this batched function is called multiple times, we do not manage h5py datasets inside.
        assert(isinstance(dataBuffer, np.ndarray))

        func = cls._get_c_dll().extract_patched_training_data_without_empty_4d_uint8

        dataSize = np.array(dataShape, dtype=np.uint64)
        patchSize = np.array(patchSize, dtype=np.uint64)
        patchStride = np.array(patchStride, dtype=np.uint64)
        patchInnerStride = np.array(patchInnerStride, dtype=np.uint64)

        patchesExtracted = ctypes.c_uint64()
        nextBatchIndex = ctypes.c_uint64()
        inputEndReached = ctypes.c_bool()

        func(dataBuffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
             ctypes.c_uint64(dataStartFlat), ctypes.c_uint64(dataEndFlat),

             dataSize.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             patchSize.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             patchStride.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             patchInnerStride.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             ctypes.c_uint64(lastFrameGap),

             ctypes.c_bool(skipEmptyPatches), ctypes.c_uint8(emptyValue),
             ctypes.c_uint64(batchStartIndex), ctypes.c_uint64(batchSize),
             ctypes.c_float(undersamplingProb),

             outputX.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
             outputY.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
             outputIndices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             ctypes.byref(patchesExtracted),
             ctypes.byref(nextBatchIndex),
             ctypes.byref(inputEndReached)
             )

        if nextBatchIndex.value == batchStartIndex:
            raise RuntimeError("No patches have been processed, does the input buffer contain the next patch?")

        return patchesExtracted.value, nextBatchIndex.value, inputEndReached.value

    @classmethod
    def extract_patched_training_data_without_empty_4d_multi(
                                                       cls, dataBuffer: np.ndarray, dataShape: Tuple,
                                                       dataStartFlat: int, dataEndFlat: int, outputX: np.ndarray,
                                                       outputY: np.ndarray, outputIndices: np.ndarray, patchSize: Tuple,
                                                       patchStride: Tuple, batchStartIndex: int, batchSize: int,
                                                       patchInnerStride: Tuple = (1, 1, 1, 1), lastFrameGap: int = 1,
                                                       undersamplingProb: float = 1.0,
                                                       skipEmptyPatches: bool = True, emptyValue: int = 0,
                                                       threadNumber: int = 8):
        assert (len(dataShape) == 4)
        # C++ code copies data column-by-column along the last axis,
        # this means that the last axis must be continuous.
        assert (patchInnerStride[-1] == 1)
        assert (dataBuffer.dtype == np.uint8)
        assert (outputIndices.dtype == np.uint64)
        # Because this batched function is called multiple times, we do not manage h5py datasets inside.
        assert (isinstance(dataBuffer, np.ndarray))

        func = cls._get_c_dll().extract_patched_training_data_without_empty_4d_multithreaded_uint8

        dataSize = np.array(dataShape, dtype=np.uint64)
        patchSize = np.array(patchSize, dtype=np.uint64)
        patchStride = np.array(patchStride, dtype=np.uint64)
        patchInnerStride = np.array(patchInnerStride, dtype=np.uint64)

        patchesExtracted = ctypes.c_uint64()
        nextBatchIndex = ctypes.c_uint64()
        inputEndReached = ctypes.c_bool()

        func(dataBuffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
             ctypes.c_uint64(dataStartFlat), ctypes.c_uint64(dataEndFlat),

             dataSize.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             patchSize.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             patchStride.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             patchInnerStride.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             ctypes.c_uint64(lastFrameGap),

             ctypes.c_bool(skipEmptyPatches), ctypes.c_uint8(emptyValue),
             ctypes.c_uint64(batchStartIndex), ctypes.c_uint64(batchSize),
             ctypes.c_float(undersamplingProb),
             ctypes.c_uint64(threadNumber),


             outputX.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
             outputY.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
             outputIndices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             ctypes.byref(patchesExtracted),
             ctypes.byref(nextBatchIndex),
             ctypes.byref(inputEndReached)
             )

        if nextBatchIndex.value == batchStartIndex:
            raise RuntimeError("No patches have been processed, does the input buffer contain the next patch?")

        return patchesExtracted.value, nextBatchIndex.value, inputEndReached.value

    @classmethod
    def extract_patched_training_data_4d_multithreaded(cls, dataBuffer: np.ndarray, dataShape: Tuple, dataStartFlat: int,
                                                       dataEndFlat: int, outputX: np.ndarray, outputY: np.ndarray,
                                                       patchSize: Tuple, batchStartIndex: int, batchSize: int,
                                                       patchInnerStride: Tuple = (1, 1, 1, 1), lastFrameGap: int = 1,
                                                       threadNumber: int = 8):
        assert (len(dataShape) == 4)
        # C++ code copies data column-by-column along the last axis,
        # this means that the last axis must be continuous.
        assert (patchInnerStride[-1] == 1)
        assert (dataBuffer.dtype == np.uint8)
        # Because this batched function is called multiple times, we do not manage h5py datasets inside.
        assert (isinstance(dataBuffer, np.ndarray))

        func = cls._get_c_dll().extract_patched_training_data_multithreaded_uint8

        dataSize = np.array(dataShape, dtype=np.uint64)
        patchSize = np.array(patchSize, dtype=np.uint64)
        patchInnerStride = np.array(patchInnerStride, dtype=np.uint64)

        patchesExtracted = ctypes.c_uint64()
        nextBatchIndex = ctypes.c_uint64()
        inputEndReached = ctypes.c_bool()

        func(
            dataBuffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_uint64(dataStartFlat),
            ctypes.c_uint64(dataEndFlat),
            dataSize.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            patchSize.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            patchInnerStride.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_uint64(lastFrameGap),
            ctypes.c_uint64(batchStartIndex),
            ctypes.c_uint64(batchSize),
            ctypes.c_uint64(threadNumber),
            outputX.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            outputY.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.byref(patchesExtracted),
            ctypes.byref(nextBatchIndex),
            ctypes.byref(inputEndReached)
             )

        if nextBatchIndex.value == batchStartIndex:
            raise RuntimeError("No patches have been processed, does the input buffer contain the next patch?")

        return patchesExtracted.value, nextBatchIndex.value, inputEndReached.value

    @classmethod
    def sparse_insert_into_bna(cls, bna: 'BufferedNdArray', indices: np.ndarray, values: np.ndarray, valueNumber: int):
        assert (indices.dtype == np.uint64)
        assert (values.dtype == np.uint8)
        assert (bna.dtype == np.uint8)

        func = cls._get_c_dll().sparse_insert_into_bna_uint8

        func(
            bna,
            indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_uint64(valueNumber)
        )

    @classmethod
    def sparse_insert_slices_into_bna(cls, bna: 'BufferedNdArray', indices: np.ndarray, slices: np.ndarray,
                                      sliceNdim: int, sliceNumber: int):
        assert (indices.dtype == np.uint64)
        assert (slices.dtype == bna.dtype)
        assert (bna.dtype == np.float32)

        func = cls._get_typed_func('sparse_insert_slices_into_bna', bna.dtype)

        func(
            bna,
            cls._ndarray_as_pointer(indices),
            cls._ndarray_as_pointer(slices),
            ctypes.c_uint64(sliceNdim),
            ctypes.c_uint64(sliceNumber)
        )

    @classmethod
    def sparse_insert_patches_into_bna(cls, bna: 'BufferedNdArray', indices: np.ndarray, patches: np.ndarray,
                                       patchSize: Tuple, patchNumber: int, isConstPatch: bool):
        assert (indices.dtype == np.uint64)
        assert (patches.dtype == bna.dtype)

        if isConstPatch:
            assert patches.shape[0] == 1
        else:
            assert patches.shape[0] == patchNumber

        patchSizeArray = np.asarray(patchSize, dtype=np.uint64)

        func = cls._get_typed_func('sparse_insert_patches_into_bna', bna.dtype)

        func(
            bna,
            cls._ndarray_as_pointer(indices),
            cls._ndarray_as_pointer(patches),
            cls._ndarray_as_pointer(patchSizeArray),
            ctypes.c_uint64(patchNumber),
            ctypes.c_bool(isConstPatch)
        )

    @classmethod
    def sparse_insert_const_into_bna(cls, bna: 'BufferedNdArray', indices: np.ndarray, value: int, valueNumber: int):
        func = cls._get_c_dll().sparse_insert_const_into_bna_uint8

        assert (indices.dtype == np.uint64)
        assert (bna.dtype == np.uint8)

        func(
            bna,
            indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_uint8(value),
            ctypes.c_uint64(valueNumber)
        )

    @classmethod
    def smooth_3d_array_average(cls, data: np.ndarray, kernelRadius: int) -> np.ndarray:
        assert(data.dtype == np.float32)

        func = cls._get_c_dll().smooth_3d_array_average_float

        shape = np.asarray(data.shape, dtype=np.uint64)
        output = np.empty_like(data)
        func(
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            shape.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_uint64(kernelRadius),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

        return output

    @classmethod
    def aggregate_attention_volume(cls, attentionVolumeFull: 'BufferedNdArray', dataSize: Tuple, patchXSize: Tuple,
                                   predictionStride: Tuple, attentionAgg: 'BufferedNdArray'):

        assert dataSize == attentionAgg.shape
        assert len(attentionVolumeFull.shape) == 8
        assert len(attentionAgg.shape) == 4
        assert attentionVolumeFull.dtype == np.float32
        assert attentionAgg.dtype == np.float32

        attVolumeSizeArr = np.array(attentionVolumeFull.shape, dtype=np.uint64)
        dataSizeArr = np.array(dataSize, dtype=np.uint64)
        patchXSizeArr = np.array(patchXSize, dtype=np.uint64)
        predictionStrideArr = np.array(predictionStride, dtype=np.uint64)

        func = cls._get_c_dll().aggregate_attention_volume
        func(
            attentionVolumeFull,
            dataSizeArr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            patchXSizeArr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            predictionStrideArr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            attentionAgg
        )

    @classmethod
    def aggregate_attention_volume_dumb(cls, attentionVolumeFull: 'BufferedNdArray', dataSize: Tuple, patchSize: Tuple,
                                        predictionDelay: int, attentionAgg: 'BufferedNdArray'):
        # todo
        raise RuntimeError("This function is deprecated and should be removed.")

        assert dataSize == attentionAgg.shape
        assert len(attentionVolumeFull.shape) == 8
        assert len(attentionAgg.shape) == 4
        assert attentionVolumeFull.dtype == np.float32
        assert attentionAgg.dtype == np.float32

        attVolumeSizeArr = np.array(attentionVolumeFull.shape, dtype=np.uint64)
        dataSizeArr = np.array(dataSize, dtype=np.uint64)
        patchSizeArr = np.array(patchSize, dtype=np.uint64)

        func = cls._get_c_dll().aggregate_attention_volume_dumb
        func(
            attentionVolumeFull,
            dataSizeArr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            patchSizeArr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_uint64(predictionDelay),
            attentionAgg
        )

    @classmethod
    def aggregate_attention_volume_local_attention(cls, attentionVolumeFull: 'BufferedNdArray',
                                                   outAttentionAvg: np.ndarray, outAttentionVar: np.ndarray):

        assert len(attentionVolumeFull.shape) == 8
        assert attentionVolumeFull.dtype == np.float32
        assert outAttentionAvg.dtype == np.float64
        assert outAttentionVar.dtype == np.float64
        assert outAttentionAvg.shape == outAttentionVar.shape == attentionVolumeFull.shape[4:]

        func = cls._get_c_dll().aggregate_attention_volume_local_attention
        func(
            attentionVolumeFull,
            cls._ndarray_as_pointer(outAttentionAvg),
            cls._ndarray_as_pointer(outAttentionVar)
        )

    @classmethod
    def apply_tf_to_volume_uint8(cls, data: np.ndarray, tfArray: np.ndarray, outImages: np.ndarray):
        assert data.dtype == np.uint8
        assert len(data.shape) == 4
        assert tfArray.shape == (256, 4)
        assert tfArray.dtype == np.uint8
        assert outImages.dtype == np.uint8

        dataSizeArray = np.asarray(data.shape, dtype=np.uint64)

        func = cls._get_c_dll().apply_tf_to_volume_uint8
        func(
            cls._ndarray_as_pointer(data),
            cls._ndarray_as_pointer(dataSizeArray),
            cls._ndarray_as_pointer(tfArray),
            cls._ndarray_as_pointer(outImages)
        )

    # -------------  BufferedNdArray functions (BNA) -------------

    @classmethod
    def bna_construct(cls, filepath: str, fileMode: int, shape: Tuple, dtype: np.dtype, maxBufferSize: int):

        func = cls._get_typed_func('bna_construct', dtype)
        func.restype = ctypes.c_void_p

        shapeArray = np.asarray(shape, dtype=np.uint64)

        bnaPointer = func(
            filepath,
            ctypes.c_uint8(fileMode),
            shapeArray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_uint64(len(shape)),
            ctypes.c_uint64(maxBufferSize)
        )

        return bnaPointer

    @classmethod
    def bna_destruct(cls, bna: 'BufferedNdArray'):
        func = cls._get_typed_func('bna_destruct', bna.dtype)
        func(bna)

    @classmethod
    def bna_set_direct_mode(cls, bna: 'BufferedNdArray', isDirectMode: bool):
        func = cls._get_typed_func('bna_set_direct_mode', bna.dtype)
        func(bna, ctypes.c_bool(isDirectMode))

    @classmethod
    def bna_flush(cls, bna: 'BufferedNdArray', flushOsBuffer: bool):
        func = cls._get_typed_func('bna_flush', bna.dtype)
        func(bna, ctypes.c_bool(flushOsBuffer))

    @classmethod
    def bna_read(cls, bna: 'BufferedNdArray', index: Tuple) -> float:
        indexArray = np.asarray(index, dtype=np.uint64)
        func = cls._get_typed_func('bna_read', bna.dtype)
        func.restype = cls._dtype_to_ctypes(bna.dtype)

        return func(bna, cls._ndarray_as_pointer(indexArray))

    @classmethod
    def bna_write(cls, bna: 'BufferedNdArray', index: Tuple, value: float):
        indexArray = np.asarray(index, dtype=np.uint64)
        func = cls._get_typed_func('bna_write', bna.dtype)
        func(bna,
             cls._ndarray_as_pointer(indexArray),
             cls._dtype_to_ctypes(bna.dtype)(value))

    @classmethod
    def bna_read_slice(cls, bna: 'BufferedNdArray', sliceIndex: Tuple, sliceNdim: int, outputBuffer: np.ndarray):
        assert bna.dtype == outputBuffer.dtype

        sliceIndexArray = np.asarray(sliceIndex, dtype=np.uint64)
        func = cls._get_typed_func('bna_read_slice', bna.dtype)
        func(bna,
             cls._ndarray_as_pointer(sliceIndexArray),
             ctypes.c_uint64(sliceNdim),
             cls._ndarray_as_pointer(outputBuffer))

    @classmethod
    def bna_read_slab(cls, bna: 'BufferedNdArray', indicesLow: np.ndarray, indicesHigh: np.ndarray,
                      outputBuffer: np.ndarray):

        assert bna.dtype == outputBuffer.dtype
        assert indicesLow.dtype == np.uint64

        func = cls._get_typed_func('bna_read_slab', bna.dtype)
        func(bna,
             cls._ndarray_as_pointer(indicesLow),
             cls._ndarray_as_pointer(indicesHigh),
             cls._ndarray_as_pointer(outputBuffer))

    @classmethod
    def bna_write_slice(cls, bna: 'BufferedNdArray', sliceIndex: Tuple, sliceNdim: int, inputBuffer: np.ndarray):
        assert bna.dtype == inputBuffer.dtype

        sliceIndexArray = np.asarray(sliceIndex, dtype=np.uint64)
        func = cls._get_typed_func('bna_write_slice', bna.dtype)
        func(bna,
             cls._ndarray_as_pointer(sliceIndexArray),
             ctypes.c_uint64(sliceNdim),
             cls._ndarray_as_pointer(inputBuffer))

    @classmethod
    def bna_write_full(cls, bna: 'BufferedNdArray', inputBuffer: np.ndarray):
        assert bna.dtype == inputBuffer.dtype

        func = cls._get_typed_func('bna_write_full', bna.dtype)
        func(bna,
             cls._ndarray_as_pointer(inputBuffer))

    @classmethod
    def bna_fill_box(cls, bna: 'BufferedNdArray', value, cornerLow: Tuple, cornerHigh: Tuple):

        cornerLowArray = np.asarray(cornerLow, dtype=np.uint64)
        cornerHighArray = np.asarray(cornerHigh, dtype=np.uint64)

        func = cls._get_typed_func('bna_fill_box', bna.dtype)
        func(bna,
             cls._dtype_to_ctypes(bna.dtype)(value),
             cls._ndarray_as_pointer(cornerLowArray),
             cls._ndarray_as_pointer(cornerHighArray))

    @classmethod
    def bna_compute_buffer_efficiency(cls, bna: 'BufferedNdArray'):
        func = cls._get_typed_func('bna_compute_buffer_efficiency', bna.dtype)
        func.restype = ctypes.c_float

        return func(bna)

    @classmethod
    def bna_reset_counters(cls, bna: 'BufferedNdArray'):
        func = cls._get_typed_func('bna_reset_counters', bna.dtype)
        func(bna)

    # -------------  MtPatchProvider functions (MTPE) -------------
    @classmethod
    def mtpe_construct(cls,
                       volumeData: 'BufferedNdArray',
                       volumeSize: Tuple, featureNumber: int,
                       patchXSize: Tuple, patchYSize: Tuple, patchStride: Tuple, patchInnerStride: Tuple,
                       predictionDelay: int,
                       detectEmptyPatches: bool, emptyValue: float, emptyCheckFeature: int,
                       undersamplingProbAny: float, undersamplingProbEmpty: float, undersamplingProbNonempty: float,
                       inputBufferSize: int, threadNumber: int = 8
                       ):
        func = cls._get_typed_func('mtpe_construct', volumeData.dtype)
        func.restype = ctypes.c_void_p

        volumeSizeArray = np.asarray(volumeSize, dtype=np.uint64)
        patchXSizeArray = np.asarray(patchXSize, dtype=np.uint64)
        patchYSizeArray = np.asarray(patchYSize, dtype=np.uint64)
        patchStrideArray = np.asarray(patchStride, dtype=np.uint64)
        patchInnerStrideArray = np.asarray(patchInnerStride, dtype=np.uint64)
        mtpePointer = func(
            volumeData,
            cls._ndarray_as_pointer(volumeSizeArray),
            ctypes.c_uint64(featureNumber),
            cls._ndarray_as_pointer(patchXSizeArray),
            cls._ndarray_as_pointer(patchYSizeArray),
            cls._ndarray_as_pointer(patchStrideArray),
            cls._ndarray_as_pointer(patchInnerStrideArray),
            ctypes.c_uint64(predictionDelay),
            ctypes.c_bool(detectEmptyPatches),
            cls._dtype_to_ctypes(volumeData.dtype)(emptyValue),
            ctypes.c_uint64(emptyCheckFeature),
            ctypes.c_float(undersamplingProbAny),
            ctypes.c_float(undersamplingProbEmpty),
            ctypes.c_float(undersamplingProbNonempty),

            ctypes.c_uint64(inputBufferSize),
            ctypes.c_uint64(threadNumber)
        )

        return mtpePointer

    @classmethod
    def mtpe_extract_batch(cls,
                           mtpe: 'MtPatchExtractor',
                           batchStartIndex: int,
                           batchSize: int,
                           outX: np.ndarray,
                           outY: np.ndarray,
                           outIndices: np.ndarray):

        assert outX.dtype == mtpe.dtype
        assert outY.dtype == mtpe.dtype
        assert outIndices.dtype == np.uint64

        patchesChecked = ctypes.c_uint64()
        patchesEmpty = ctypes.c_uint64()
        patchesExtracted = ctypes.c_uint64()
        nextBatchIndex = ctypes.c_uint64()
        inputEndReached = ctypes.c_bool()

        func = cls._get_typed_func('mtpe_extract_batch', mtpe.dtype)
        func(
            mtpe,
            ctypes.c_uint64(batchStartIndex),
            ctypes.c_uint64(batchSize),
            cls._ndarray_as_pointer(outX),
            cls._ndarray_as_pointer(outY),
            cls._ndarray_as_pointer(outIndices),

            ctypes.byref(patchesChecked),
            ctypes.byref(patchesEmpty),
            ctypes.byref(patchesExtracted),
            ctypes.byref(nextBatchIndex),
            ctypes.byref(inputEndReached)
        )

        return nextBatchIndex.value, inputEndReached.value, \
               patchesChecked.value, patchesEmpty.value, patchesExtracted.value

    @classmethod
    def mtpe_destruct(cls, mtpe: 'MtPatchExtractor'):
        func = cls._get_typed_func('mtpe_destruct', mtpe.dtype)
        func(mtpe)

    @classmethod
    def _get_typed_func(cls, funcName, dtype):
        func = getattr(cls._get_c_dll(),
                       '{funcName}_{dtype}'.format(funcName=funcName, dtype=cls._get_dtype_name(dtype)))
        return func

    @classmethod
    def _get_dtype_name(cls, dtype: np.dtype) -> str:
        if dtype == np.dtype(np.uint8):
            return 'uint8'
        elif dtype == np.dtype(np.uint64):
            return 'uint64'
        elif dtype == np.dtype(np.float32):
            return 'float32'
        else:
            raise RuntimeError("Unsupported dtype: '{}'".format(dtype.str))

    @classmethod
    def _ndarray_as_pointer(cls, array: np.ndarray):
        cls._check_buffer_continuous(array)
        return array.ctypes.data_as(ctypes.POINTER(cls._dtype_to_ctypes(array.dtype)))

    @classmethod
    def _check_buffer_continuous(cls, buffer: np.ndarray) -> np.ndarray:
        """
        If a noncontinuous numpy array is given for reading,
        then create a continuous copy of the array and warn the user.
        """
        if not buffer.flags['C_CONTIGUOUS']:
            warnings.warn("The given array is not a continuous buffer. "
                          "This can lead to incorrect reads. Forcing a copy.")
            return buffer.copy()
        return buffer

    @classmethod
    def _dtype_to_ctypes(cls, dtype: np.dtype) -> Type:
        if dtype == np.uint8:
            return ctypes.c_uint8
        elif dtype == np.uint64:
            return ctypes.c_uint64
        elif dtype == np.float32:
            return ctypes.c_float
        elif dtype == np.float64:
            return ctypes.c_double
        else:
            raise TypeError("Unsupported dtype: {}".format(dtype.str))

    @classmethod
    def _compute_patch_number(cls, dataShape: Tuple, sourceAxes: Tuple, patchSize: Tuple,
                              patchStride: Tuple = None):
        patchNumber = []
        for i, axis in enumerate(sourceAxes):
            totalPatchNumber = dataShape[axis] - patchSize[sourceAxes.index(axis)] + 1
            stride = patchStride[i]
            patchNumber.append(int(math.ceil(totalPatchNumber / stride)))

        return patchNumber

    @classmethod
    def _compute_patch_shape(cls, dataShape: Tuple, sourceAxes: Tuple, patchSize: Tuple):
        patchShape = []
        for axis in range(len(dataShape)):
            patchShape.append(patchSize[sourceAxes.index(axis)] if axis in sourceAxes else dataShape[axis])
        return patchShape