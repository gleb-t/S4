import glob
import shutil
import math
import os
import os.path as path
from pathlib import Path
import warnings
import json
import gzip
from enum import IntEnum

from typing import *

import numpy as np
import scipy.ndimage
import scipy.misc
import h5py
from deprecation import deprecated

import PythonExtras.numpy_extras as npe
from PythonExtras import file_tools
from PythonExtras.BufferedNdArray import BufferedNdArray
from PythonExtras.CppWrapper import CppWrapper

T = TypeVar('T')
TTFFunc = Callable[[int], Tuple[float, float, float, float]]

class DataMissingError(Exception):
    """
    Thrown when a metadata file points to non-existent data.
    """


class DataBrokenError(Exception):
    """
    Thrown when failing to load the data from disk. E.g. metadata doesn't match the data.
    """


class VolumeMetadata:
    class Compression(IntEnum):
        none = 0
        gzip = 1

    class Format(IntEnum):
        datRaw = 0
        datBna = 1

    # Order in which the metadata is parsed. Important for some consistency checks.
    _parseOrder = [
        'ContainerFormat'.lower(),
        'ComponentNumber'.lower(),
        'Resolution'.lower(),
        'ObjectFileName'.lower(),
        'FrameNumber'.lower(),
        'Compression'.lower(),
        'Format'.lower(),
        'EmptyValues'.lower()
    ]

    def __init__(self, format=Format.datRaw, isTemporal=False, baseDirPath='', dataPath='',
                 spatialShape=None, frameNumber=0, dtype=np.uint8,
                 componentNumber=1, compression=Compression.none,
                 emptyValues: Union[np.ndarray, List[float]]=None):

        self.format = format  # type: VolumeMetadata.Format
        self.isTemporal = isTemporal  # type: bool
        self.baseDirPath = baseDirPath
        self.dataPath = dataPath
        self.dataFileExtension = None  # type: Union[str, None]
        self.filepaths = []  # type: List[str]
        self.spatialShape = spatialShape  # type: Tuple[int, int, int]
        self.frameNumber = frameNumber
        self.dtype = np.dtype(dtype)
        self.componentNumber = componentNumber
        self.compression = compression
        self.emptyValues = None if emptyValues is None else [self.dtype.type(x) for x in emptyValues]

        if self.emptyValues is not None and len(self.emptyValues) != self.componentNumber:
            raise RuntimeError("Invalid empty value number: {}, expected {}."
                               .format(len(self.emptyValues), self.componentNumber))

    # todo convert to a property, for consistency.
    def get_shape(self, forceMultivar: bool = False) -> Tuple[int, ...]:
        shapeRaw = []
        if self.isTemporal:
            shapeRaw.append(self.frameNumber)

        shapeRaw += list(self.spatialShape)

        if self.componentNumber > 1 or forceMultivar:
            shapeRaw.append(self.componentNumber)

        return tuple(shapeRaw)

    def filter_timesteps(self, timeSlice: slice = None, timestepsToLoad: List[int] = None):
        # todo Filtering the timesteps shouldn't be the metadata class'es responsibility. Let the loader handle that.
        assert self.frameNumber > 0

        if self.format == VolumeMetadata.Format.datRaw:
            if timestepsToLoad is not None:
                self.filepaths = [self.filepaths[i] for i in timestepsToLoad]
            if timeSlice is not None:
                self.filepaths = [self.filepaths[i] for i in range(*timeSlice.indices(self.frameNumber))]

            self.frameNumber = len(self.filepaths)
        elif self.format == VolumeMetadata.Format.datBna:
            if timestepsToLoad is not None or (timeSlice is not None and timeSlice != slice(None)):
                raise NotImplementedError("Filtering datBna volumes by time is not currently supported.")

        return self

    @classmethod
    def load_from_dat(cls, datPath: str) -> 'VolumeMetadata':
        if not os.path.exists(datPath):
            raise RuntimeError("Path does not exist: '{}'".format(datPath))

        meta = VolumeMetadata()

        meta.baseDirPath = os.path.dirname(datPath)

        filename, extension = os.path.splitext(datPath)
        if not extension.lower() == '.dat':
            raise RuntimeError("Path does no point to a .dat file: '{}'".format(datPath))

        for key, valueString in cls._read_dat_sorted(datPath):
            if key == 'ContainerFormat'.lower():
                try:
                    meta.format = VolumeMetadata.Format[valueString]
                except KeyError:
                    raise RuntimeError("Unknown volume (container) format: '{}'".format(valueString))
            elif key == 'ComponentNumber'.lower():
                meta.componentNumber = int(valueString)
            elif key == 'Resolution'.lower():
                # e.g. "128 256 256" -> (256, 256, 128)
                meta.spatialShape = tuple(reversed([int(c.strip()) for c in valueString.split()]))
            elif key == 'ObjectFileName'.lower():
                # meta.dataPath = os.path.join(meta.baseDirPath)
                targetFilesString = valueString

                # The 'ObjectFileName' parameter specifies either the dir where all files should be loaded,
                # or a wildcard pattern, matching files with a certain extension, e.g. "/path/to/dir/*.raw".
                starIndex = targetFilesString.find('*')
                if starIndex > 0:
                    if targetFilesString[starIndex + 1] != '.':
                        raise RuntimeError("Invalid volume wildcard provided: '{}'".format(targetFilesString))

                    meta.dataFileExtension = targetFilesString[starIndex + 1:]
                    targetFilesString = targetFilesString[:starIndex].replace('\\', os.path.sep)

                meta.dataPath = os.path.join(meta.baseDirPath, targetFilesString)

                if not os.path.exists(meta.dataPath):
                    raise DataMissingError("Target path doesn't exist: '{}' => '{}'"
                                           .format(valueString, meta.dataPath))

                if os.path.isdir(meta.dataPath):
                    meta.isTemporal = True

                    filenames = [f for f in sorted(os.listdir(meta.dataPath))]
                    if meta.dataFileExtension is not None:
                        filenames = [f for f in filenames if os.path.splitext(f)[1] == meta.dataFileExtension]

                    meta.filepaths = [os.path.join(meta.dataPath, f) for f in filenames]
                    meta.frameNumber = len(meta.filepaths)
                else:
                    meta.filepaths = [meta.dataPath]
                    if meta.format == VolumeMetadata.Format.datRaw:
                        meta.frameNumber = 1
                    # If a 'datBna' file is provided, it must specify the 'FrameNumber' property.
            elif key == 'FrameNumber'.lower():
                value = int(valueString)
                if meta.format == VolumeMetadata.Format.datRaw:
                    if meta.frameNumber != 0 and meta.frameNumber != value:
                        raise RuntimeError("Specified frame number ({}) doesn't match "
                                           "the number of volume files ({})".format(value, meta.frameNumber))
                elif meta.format == VolumeMetadata.Format.datBna:
                    # For 'datBna' volumes, the frame number is always specified.
                    if value > 0:
                        meta.isTemporal = True
                    meta.frameNumber = value
                else:
                    raise ValueError("Unknown volume format: '{}'".format(meta.format))

            elif key == 'Compression'.lower():
                if valueString.lower() == 'gzip':
                    meta.compression = VolumeMetadata.Compression.gzip
                elif valueString.lower() == 'none':
                    meta.compression = VolumeMetadata.Compression.none
                else:
                    raise RuntimeError("Unknown compression type: '{}'".format(valueString))
            elif key == 'Format'.lower():
                if valueString.lower() == 'uchar' or valueString.lower() == 'uint8':
                    meta.dtype = np.dtype(np.uint8)
                elif valueString.lower() == 'float' or valueString.lower() == 'float32':
                    meta.dtype = np.dtype(np.float32)
                else:
                    raise RuntimeError("Unsupported volume data type: '{}'".format(valueString))
            elif key == 'EmptyValues'.lower():
                values = [meta.dtype.type(x) for x in valueString.split(',')]
                if len(values) != meta.componentNumber:
                    raise RuntimeError("Invalid empty value number: {}, expected {}."
                                       .format(len(values), meta.componentNumber))

                meta.emptyValues = values
            else:
                warnings.warn("Skipped an unknown .dat parameter: '{}'".format(key))

        return meta

    @classmethod
    def _read_dat_sorted(cls, datPath: str):
        props = []
        with open(datPath, 'r') as file:
            for line in file.readlines():
                if len(line.strip()) == 0:
                    continue

                chunks = [c.strip() for c in line.split(':')]

                props.append((chunks[0].lower(), chunks[1]))

        return sorted(props, key=lambda item: cls._parseOrder.index(item[0]) if item[0] in cls._parseOrder else 0)

    def write_to_dat(self, datPath: str):
        volumeSizeString = ' '.join(reversed([str(x) for x in self.spatialShape[0:3]]))  # Specified as X Y Z

        relDataPath = os.path.relpath(self.dataPath, os.path.dirname(datPath))
        with open(datPath, 'w', encoding='ascii') as file:
            file.write('ObjectFileName: {}\n'.format(relDataPath))
            file.write('ContainerFormat: {}\n'.format(self.format.name))
            file.write('Resolution: {}\n'.format(volumeSizeString))
            file.write('Format: {}\n'.format(_dtype_to_string(self.dtype)))
            file.write('ComponentNumber: {}\n'.format(self.componentNumber))
            file.write('Compression: {}\n'.format(self.compression.name))
            if self.isTemporal:
                file.write('FrameNumber: {}\n'.format(self.frameNumber))
            if self.emptyValues is not None:
                file.write('EmptyValues: {}\n'.format(','.join((str(x) for x in self.emptyValues))))


def load_volume_data_from_dat_with_meta(datPath,
                                        outputAllocator: Callable[[Tuple, Type], npe.LargeArray] = None,
                                        timestepsToLoad: List[int] = None,
                                        timeSlice: slice = None,
                                        forceMultivar: bool = False,
                                        printFn: Callable[[Any], None] = None) -> Tuple[npe.LargeArray, VolumeMetadata]:
    metadata = VolumeMetadata.load_from_dat(datPath)
    data = load_volume_data_from_metadata(metadata, outputAllocator=outputAllocator,
                                          timestepsToLoad=timestepsToLoad, timeSlice=timeSlice,
                                          forceMultivar=forceMultivar, printFn=printFn)

    return data, metadata


def load_volume_data_from_dat(datPath,
                              outputAllocator: Callable[[Tuple, Type], npe.LargeArray] = None,
                              timestepsToLoad: List[int] = None,
                              timeSlice: slice = None,
                              forceMultivar: bool = False,
                              printFn: Callable[[Any], None] = None):
    """
    Reads a provided .dat metadata file, and loads a corresponding volume or a volume sequence.
    Use the allocator callback to flexibly define how the output show be stored.
    """

    metadata = VolumeMetadata.load_from_dat(datPath)
    return load_volume_data_from_metadata(metadata, outputAllocator=outputAllocator,
                                          timestepsToLoad=timestepsToLoad, timeSlice=timeSlice,
                                          forceMultivar=forceMultivar, printFn=printFn)


def load_volume_data_from_metadata(metadata: VolumeMetadata,
                                   outputAllocator: Callable[[Tuple, Type], npe.LargeArray] = None,
                                   timestepsToLoad: List[int] = None,
                                   timeSlice: slice = None,
                                   forceMultivar: bool = False,
                                   printFn: Callable[[Any], None] = None) -> npe.LargeArray:
    """
    Use this method to load volume data when no .dat file exists and metadata needs to be provided manually.
    """
    # If we don't have a sequence, parse a single static volume.
    if not metadata.isTemporal:

        fileData = _read_binary_file(metadata.filepaths[0], metadata.dtype, metadata.compression)

        if metadata.componentNumber == 1 and not forceMultivar:
            targetShape = metadata.spatialShape
        else:
            targetShape = metadata.spatialShape + (metadata.componentNumber,)

        if outputAllocator is not None:
            data = outputAllocator(targetShape, metadata.dtype)
            data[...] = fileData.reshape(targetShape)
            return data
        else:
            return fileData

    if outputAllocator is None:
        outputAllocator = lambda shapeToAllocate, dtype: np.zeros(shapeToAllocate, dtype)

    if (timeSlice is not None and timeSlice != slice(None)) or bool(timestepsToLoad):
        metadata.filter_timesteps(timeSlice, timestepsToLoad)

    shape = metadata.get_shape()
    if metadata.componentNumber == 1 and forceMultivar:
        shape = shape + (1,)

    data = outputAllocator(shape, metadata.dtype)
    for iFile, filepath in enumerate(metadata.filepaths):
        if printFn:
            printFn("Reading file {}".format(filepath))

        try:
            fileData = _read_binary_file(filepath, metadata.dtype, metadata.compression)

            if metadata.format == VolumeMetadata.Format.datRaw:
                # The 'datRaw' data is stored frame-by-frame in separate files.
                data[iFile, ...] = fileData.reshape(shape[1:])
            elif metadata.format == VolumeMetadata.Format.datBna:
                # The 'datBna' data is stored all together in a single file.
                # (Still load frame-by-frame to reduce RAM usage.)
                fileData = fileData.reshape(shape)
                for f in range(metadata.frameNumber):
                    data[f, ...] = fileData[f, ...]
            else:
                raise ValueError("Unknown volume format: {}".format(metadata.format))
        except Exception as e:
            raise DataBrokenError("Failed to load data from '{}': '{}'".format(filepath, e)) from e

    return data


def downsample_volume_sequence(inputDatPath, outputDatPath, outputShape, printFn=None):
    inputData = load_volume_data_from_dat(inputDatPath, printFn=None)

    filename, extension = os.path.splitext(os.path.basename(outputDatPath))
    dirPath = os.path.join(os.path.dirname(outputDatPath), filename)
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

    for f in range(0, inputData.shape[0]):
        if printFn:
            printFn("Downsampling frame {}/{}".format(f, inputData.shape[0]))

        outputFrame = downsample_volume(inputData[f, ...], outputShape)
        filepath = os.path.join(dirPath, '{:06d}.raw'.format(f))
        outputFrame.tofile(filepath)

    _write_metadata_to_dat(outputDatPath, outputShape, filename, inputData.dtype)


def resize_array_point(data, outputSize):
    """
    Resizes an Nd array using point (nearest neighboor sampling).
    :return:
    """

    return CppWrapper.resize_array_point(data, outputSize)


def downsample_volume(data, outputSize):
    inputSize = data.shape

    if inputSize == outputSize:
        return data

    assert(data.ndim == 3)
    assert(data.ndim == len(outputSize))

    resizeRatio = max(np.asarray(inputSize) / np.asarray(outputSize))

    kernelRadius = int(math.ceil(resizeRatio / 2))
    kernelWidth = 2 * kernelRadius + 1

    # Fit full Gaussian into the kernel.
    sigma = kernelWidth / 6.0

    # Precompute kernel values. No scaling constant, since we normalize anyway.
    kernel = np.zeros(kernelWidth)
    kernel[kernelRadius] = 1.0  # middle
    sigmaSqr = sigma ** 2
    sum = 1.0
    for i in range(1, kernelRadius + 1):
        w = math.exp(-0.5 * float(i ** 2) / sigmaSqr)
        kernel[kernelRadius + i] = w
        kernel[kernelRadius - i] = w
        sum += 2.0 * w

    for i in range(0, kernelWidth):
        kernel[i] /= sum

    input = data
    output = None
    for axis in range(0, data.ndim):
        output = scipy.ndimage.convolve1d(input, kernel, axis=axis, mode='mirror')
        if axis != data.ndim - 1:
            input = output.copy()

    return resize_array_point(output, outputSize)


def downsample_shape(shape: Tuple[T, ...],
                     volumeCrop: Optional[Tuple[Optional[slice], ...]] = None,
                     downsampleFactors: Optional[Sequence[float]] = None) -> Tuple[T, ...]:

    if volumeCrop is not None:
        shape = npe.slice_shape(shape, volumeCrop)
    if downsampleFactors:
        # todo A hacky way of figuring out what the downsampled shape would be. Known to break!
        shape = tuple(int(x / f) for x, f in zip(shape, downsampleFactors))

    return shape


def get_ensemble_min_max_shape(ensemblePathPattern: str) -> Tuple[Tuple, Tuple]:
    allMemberShapes = []
    memberMetaPaths = glob.glob(ensemblePathPattern)
    for metaPath in memberMetaPaths:
        meta = VolumeMetadata.load_from_dat(metaPath)
        allMemberShapes.append(meta.get_shape())

    allShapesArray = np.asarray(allMemberShapes)

    return tuple(np.min(allShapesArray, axis=0)), tuple(np.max(allShapesArray, axis=0))


@deprecated(details="Use VolumeMetadata class instead.")
def get_volume_file_list(datPath: str) -> List[str]:

    metadata = _read_metadata_from_dat(datPath)
    return _get_volume_file_list(os.path.dirname(datPath), metadata, None)


def _get_volume_file_list(baseDirPath, metadata,
                          timestepsToLoad: List[int] = None, timeSlice: slice = None) -> List[str]:
    targetFilesString = metadata['ObjectFileName'.lower()]

    # The 'ObjectFileName' parameter specifies either the dir where all files should be loaded,
    # or a wildcard pattern, matching files with a certain extension, e.g. "/path/to/dir/*.raw".
    volumeExtensionFilter = None
    starIndex = targetFilesString.find('*')
    if starIndex > 0:
        if targetFilesString[starIndex + 1] != '.':
            raise RuntimeError("Invalid volume wildcard provided: '{}'".format(targetFilesString))

        volumeExtensionFilter = targetFilesString[starIndex + 1:]
        targetFilesString = targetFilesString[:starIndex]

    dataPath = os.path.join(baseDirPath, targetFilesString)

    # If the target path is not a dir, it's a single static volume.
    if not os.path.isdir(dataPath):
        return [dataPath]

    # Figure out exactly which files to load.
    filenames = [f for f in os.listdir(dataPath)]
    if volumeExtensionFilter is not None:
        filenames = [f for f in filenames if os.path.splitext(f)[1] == volumeExtensionFilter]
    if timestepsToLoad is not None:
        filenames = [filenames[f] for f in timestepsToLoad]
    if timeSlice is not None:
        filenames = [filenames[f] for f in range(*timeSlice.indices(len(filenames)))]  # Apply the slicing to the list.

    return [os.path.join(dataPath, f) for f in sorted(filenames)]


def _read_binary_file(volumeFilePath: str, dtype: np.dtype,
                      compression: 'VolumeMetadata.Compression' = VolumeMetadata.Compression.none) -> np.ndarray:

    if compression == VolumeMetadata.Compression.none and os.path.splitext(volumeFilePath)[1] == '.gz':
        warnings.warn("Reading '.gz' volume file as uncompressed. Is the metadata outdated?")

    if compression == VolumeMetadata.Compression.none:
        with open(volumeFilePath, 'rb') as file:
            frameData = np.fromfile(file, dtype)
    elif compression == VolumeMetadata.Compression.gzip:
        with gzip.open(volumeFilePath, 'rb') as file:
            frameData = np.frombuffer(file.read(), dtype=dtype)
    else:
        raise RuntimeError("Unknown compression method: '{}'".format(str(compression)))

    return frameData


@deprecated(details="Use VolumeMetadata class instead.")
def read_metadata_from_dat(datPath):
    return _read_metadata_from_dat(datPath)


def _read_metadata_from_dat(datPath):

    if not os.path.exists(datPath):
        raise RuntimeError("Path does not exist: '{}'".format(datPath))

    filename, extension = os.path.splitext(datPath)
    if not extension.lower() == '.dat':
        raise RuntimeError("Path does no point to a .dat file: '{}'".format(datPath))

    data = {}

    with open(datPath, 'r') as file:
        for line in file.readlines():
            chunks = [c.strip() for c in line.split(':')]

            # Parse as a string by default.
            key, valueString = chunks[0].lower(), chunks[1]
            # Convert known numeric parameters to integers.
            if key == 'ComponentNumber'.lower():
                value = int(valueString)
            elif key == 'Resolution'.lower():
                value = tuple([int(c.strip()) for c in valueString.split()])  # e.g. "256 256 256"
            else:
                value = valueString.strip()

            data[key] = value

    return data


@deprecated(details="Use 'load_volume_from_dat instead")
def load_volume_sequence_from_dir(dir, size, frameSlice, preloadedDataPath=None, printFn=None):

    filenames = [f for f in os.listdir(dir)]
    width, height, depth = size

    if not os.path.isfile(preloadedDataPath):
        if printFn:
            printFn("Reading data from {}".format(dir))

        size4d = (len(filenames),) + size
        data = np.empty(size4d, dtype=np.uint8)
        for i, filename in enumerate(filenames):
            if printFn:
                printFn("Reading file {}".format(filename))

            filepath = os.path.join(dir, filename)
            with open(filepath, 'rb') as file:
                frameData = np.fromfile(file, np.uint8, width * height * depth)

            frameData = frameData.reshape((depth, height, width))
            data[i, ...] = frameData

        h5File = h5py.File(preloadedDataPath, 'w')
        h5Data = h5File.create_dataset('volume-data', data=data, dtype='uint8')
        h5Data[...] = data
        data = data[frameSlice, ...]

    else:
        if printFn:
            printFn("Reading preloaded data from {}".format(preloadedDataPath))
        h5File = h5py.File(preloadedDataPath, 'r')
        data = h5File['volume-data'][frameSlice, ...]

    return data


def write_volume_sequence(inputPath: str,
                          data: Union[np.ndarray, h5py.Dataset, BufferedNdArray],
                          dataFormat: VolumeMetadata.Format = VolumeMetadata.Format.datRaw,
                          timeAxis: int = 0,
                          filenameProvider: Callable[[int], str] = None,
                          clip: Tuple = None,
                          dtype: Type = None,
                          compress: bool = False,
                          emptyValues: Union[np.ndarray, List[float]] = None,
                          clearDir: bool = True,
                          printFn: Callable[[Any], None] = None):

    assert(timeAxis == 0)  # For now just assume that time is the first axis.

    if filenameProvider is None:
        filenameProvider = lambda x: 'frame_{:04d}.raw'.format(x)

    if dtype is None:
        dtype = data.dtype

    inputPath = Path(inputPath)
    metaPath = str(inputPath.with_suffix('.dat'))
    dataPath = str(inputPath.parent / inputPath.stem)  # Without the extension.
    basePath = str(inputPath.parent)

    if dataFormat == VolumeMetadata.Format.datRaw:
        # For .dat volumes, we might need to create and clear the data dir.
        file_tools.create_dir(dataPath)
        if clearDir:
            file_tools.clear_dir(dataPath)
    elif dataFormat == VolumeMetadata.Format.datBna:
        dataPath += '.bna'  # For BNA volumes, the data is stored in a single file not a dir.

    if printFn:
        printFn("Writing volume sequence data to '{}'".format(dataPath))

    if dataFormat == VolumeMetadata.Format.datRaw:
        for f in range(0, data.shape[timeAxis]):
            frameData = data[f, ...]
            if clip is not None:
                frameData = np.clip(frameData, clip[0], clip[1])
            frameData = frameData.astype(dtype)

            filename = filenameProvider(f)
            filepath = os.path.join(dataPath, filename)
            _write_binary_file(filepath, frameData, compress)

    elif dataFormat == VolumeMetadata.Format.datBna:
        with BufferedNdArray(dataPath, BufferedNdArray.FileMode.rewrite,
                             data.shape, dtype=dtype) as bna:
            # This is a bit of a copy-paste, but prettier than adding 'ifs' all over.
            for f in range(0, data.shape[timeAxis]):
                frameData = data[f, ...]
                if clip is not None:
                    frameData = np.clip(frameData, clip[0], clip[1])
                frameData = frameData.astype(dtype)

                bna[f] = frameData

        if compress:
            _gzip_large_file(dataPath, removeOriginal=True)
    else:
        raise ValueError("Unknown volume format.")

    if printFn:
        printFn("Writing volume sequence metadata to '{}'".format(metaPath))

    componentNumber = 1 if len(data.shape) == 4 else data.shape[4]
    compression = VolumeMetadata.Compression.gzip if compress else VolumeMetadata.Compression.none
    metadata = VolumeMetadata(format=dataFormat, isTemporal=True, baseDirPath=basePath, dataPath=dataPath,
                              spatialShape=data.shape[1:4], frameNumber=data.shape[timeAxis],
                              dtype=dtype, componentNumber=componentNumber,
                              compression=compression, emptyValues=emptyValues)
    metadata.write_to_dat(metaPath)


def write_volume_to_datraw(data: npe.LargeArray, rawOrDatPath: str, dataOrder: str = 'zyx',
                           compress: bool = False,
                           printFn: Callable[[Any], None] = None):
    """
    Datraw: raw format, with a metadata .dat file describing the resolution.

    :param data:
    :param rawOrDatPath:
    :param dataOrder:
    :param compress:
    :param printFn:
    :return:
    """

    beforeExtension, extension = path.splitext(rawOrDatPath)

    assert data.dtype == np.uint8

    if dataOrder == 'zyx':
        pass
    elif dataOrder == 'xyz':
        data = data.swapaxes(0, 2)
    else:
        raise RuntimeError("Unsupported data order: {}".format(dataOrder))

    rawPath = beforeExtension + '.raw'
    metaPath = beforeExtension + '.dat'

    if printFn:
        printFn("Writing raw volume data to '{}'".format(rawPath))

    rawPath = _write_binary_file(rawPath, data[...], compress=compress)  # Ellipsis in case a BNA was provided.

    if printFn:
        printFn("Writing volume metadata to '{}'".format(metaPath))

    compression = 'gzip' if compress else 'none'
    _write_metadata_to_dat(metaPath, data.shape, os.path.basename(rawPath), dtype=data.dtype, compression=compression)


def _write_binary_file(filepath, frameData, compress):
    if not compress:
        frameData.tofile(filepath)
    else:
        filepath += '.gz'
        with gzip.open(filepath, 'wb', compresslevel=6) as f:
            f.write(frameData.tobytes())

    return filepath


def _gzip_large_file(filepath: str, removeOriginal: bool = True):
    with open(filepath, 'rb') as fileRaw, gzip.open(filepath + '.gz', 'wb') as fileComp:
        shutil.copyfileobj(fileRaw, fileComp)

    if removeOriginal:
        os.remove(filepath)


@deprecated(details="Use VolumeMetadata class instead.")
def _write_metadata_to_dat(metaPath, shape, dataFilename, dtype: np.dtype, compression='none'):
    volumeSizeString = ' '.join(reversed([str(x) for x in shape[0:3]]))  # Specified as X Y Z
    componentNumber = 1 if len(shape) == 3 else shape[3]

    with open(metaPath, 'w', encoding='ascii') as file:
        file.write('ObjectFileName: {}\n'.format(dataFilename))
        file.write('Resolution: {}\n'.format(volumeSizeString))
        file.write('Format: {}\n'.format(_dtype_to_string(dtype)))
        file.write('ComponentNumber: {}\n'.format(componentNumber))
        file.write('Compression: {}\n'.format(compression))


def _dtype_to_string(dtype: np.dtype):
    if dtype == np.uint8:
        return 'uchar'
    elif dtype == np.float32:
        return 'float'
    else:
        raise RuntimeError("Unsupported volume data type: '{}'".format(dtype))


def write_parallax_seven_config(configPath, series: List[Dict[str, Any]]):
    for seriesDesc in series:
        if 'path' not in seriesDesc:
            raise RuntimeError("Series must specify a path.")

        if 'slice' in seriesDesc:
            sliceObject = seriesDesc['slice']
            if sliceObject and (sliceObject.start or sliceObject.stop or sliceObject.step):
                sliceString = '{},{},{}'.format(sliceObject.start, sliceObject.stop, sliceObject.step)
                seriesDesc['slice'] = sliceString
            else:
                del seriesDesc['slice']

    with open(configPath, 'w', encoding='utf-8') as file:
        json.dump({'series': series}, file)


def allocate_temp_bna(shape: Tuple, dtype: Type):
    """ Helper function to be used as an allocator for volume loading."""
    return BufferedNdArray(file_tools.get_temp_filename(),
                           BufferedNdArray.FileMode.rewrite, shape, np.dtype(dtype),
                           maxBufferSize=int(1e9), deleteOnDestruct=True)

