import os
import warnings
import ctypes
from enum import IntEnum, Enum
from typing import Callable, Tuple, List, Dict, Any, Union, Type, Iterable

import numpy as np

from PythonExtras.CppWrapper import CppWrapper


class BufferedNdArray:
    """
    A Python wrapper around a C++ class implementing buffered read/write access
    to an nd-array on disk.
    This wrapper stores a pointer to a C++ object and uses ctypes and static wrapper
    functions to call the class methods.
    """

    class FileMode(IntEnum):
        """
        The integer value here must be identical to the C++ implementation.
        """
        unknown = 0
        readonly = 1
        update = 2
        rewrite = 3

    class NoncontinuousArray(UserWarning):
        pass

    class Query:
        """
        Parses and stores the query used when indexing the array (reading or writing).
        """

        class Type(Enum):
            unknown    = 0
            cell       = 1  # A single cell of the array, (int{ndim}) query.
            slice      = 2  # Only a scalar on outer axes, full inner axes, (int+, None*) query.
            slab_outer = 3  # Similar to slice, but with a range the outer-most axis, (slice, None*) query.
            slab       = 4  # Arbitrary indices/ranges along any axes, ([int, slice, None]+) query.
            full       = 5  # Write all values in the array.

        class SlabAxisDesc:

            def __init__(self, start: int, end: int, keepAxis: bool):
                self.start = start
                self.end = end
                self.keepAxis = keepAxis

        def __init__(self, queryRaw, shape: Tuple):
            self._queryRaw = queryRaw
            self._shape = shape
            self.type = BufferedNdArray.Query.Type.unknown
            ndim = len(self._shape)

            queryNorm = queryRaw
            if not isinstance(queryRaw, tuple):
                queryNorm = (queryRaw,)

            # Check query bounds. We don't support negative indices for now.
            for v in queryNorm:
                if isinstance(v, (int, float)):
                    if v < 0 or v >= self._shape[0]:
                        raise IndexError("Index out of bounds: {}".format(queryRaw))
                elif isinstance(v, slice):
                    if v.start is not None and (v.start < 0 or v.start >= self._shape[0]):
                        raise IndexError("Index out of bounds: {}".format(queryRaw))
                    if v.stop is not None and (v.stop < 1 or v.stop > self._shape[0]):
                        raise IndexError("Index out of bounds: {}".format(queryRaw))

            if len(queryNorm) > ndim:
                raise RuntimeError("Too many query dimensions given ({}, expected up to {})".format(
                    len(queryNorm), ndim
                ))

            # Expand Ellipsis into None's.
            if Ellipsis in queryNorm:
                if sum(1 for x in queryNorm if x is Ellipsis) > 1:
                    raise RuntimeError("Too many ellipsis' in the query.")

                queryPadded = []
                # for dim in range(self._bna.ndim):
                for i, x in enumerate(queryNorm):
                    if x is Ellipsis:
                        queryPadded += [None] * (ndim - len(queryNorm) + 1)  # +1 for the Ellipsis itself.
                    else:
                        queryPadded.append(x)

                queryNorm = tuple(queryPadded)
                assert len(queryNorm) == ndim

            # Replace trivial slices by None's.
            queryNorm = tuple(map(lambda x: None if isinstance(x, slice) and x.start is None and x.stop is None else x,
                                  queryNorm))

            # Add None's for missing indices.
            if len(queryNorm) < ndim:
                queryNorm = queryNorm + (None,) * (ndim - len(queryNorm))

            # Check for a full query.
            if all((x is None for x in queryNorm)):
                self.type = BufferedNdArray.Query.Type.full

                return

            # Check for a cell-type query (int, int, int)
            if all((isinstance(x, int) and self._check_index(x, dim) for dim, x in enumerate(queryNorm))):
                self.type = BufferedNdArray.Query.Type.cell
                self.cellIndex = queryNorm  # type: Tuple

                return

            # Check for a slice-type query (int, int, None, None)
            nonIntIndex = next((i for i, x in enumerate(queryNorm) if not isinstance(x, int)))
            if all((x is None for x in queryNorm[nonIntIndex:])):
                for dim, x in enumerate(queryNorm[:nonIntIndex]):
                    self._check_index(x, dim)

                self.type = BufferedNdArray.Query.Type.slice
                self.sliceIndex = queryNorm[:nonIntIndex]
                self.sliceNdim = ndim - nonIntIndex

                return
            
            # Check for an outer slab query (slice, None, None).
            if isinstance(queryNorm[0], slice) and all((x is None for x in queryNorm[1:])):
                firstSlice = queryNorm[0]
                start = firstSlice.start if firstSlice.start is not None else 0
                end = firstSlice.stop if firstSlice.stop is not None else self._shape[0]

                # If the outer slice selects everything, the full dataset is indexed.
                if start == 0 and end == self._shape[0]:  # todo Unit test this.
                    self.type = BufferedNdArray.Query.Type.full
                    return

                self.type = BufferedNdArray.Query.Type.slab_outer
                self.slabIndex = [None] * ndim  # type: List[BufferedNdArray.Query.SlabAxisDesc]
                self.slabIndex[0] = BufferedNdArray.Query.SlabAxisDesc(start, end, True)
                for dim in range(1, ndim):
                    self.slabIndex[dim] = BufferedNdArray.Query.SlabAxisDesc(0, self._shape[dim], True)

                self.slabShape = tuple((i.end - i.start for i in self.slabIndex))

                return

            # Check for a slab-type query (int, slice, None, int, slice)
            if all((x is None or isinstance(x, int) or isinstance(x, slice) for x in queryNorm)):
                self.type = BufferedNdArray.Query.Type.slab
                # Check how many None's are at the end. This is the ndim of slices we'll use to cut out the slab.
                self.slabSubsliceNdim = next((i for i, v in enumerate(queryNorm[::-1]) if v is not None))  # type: int

                # If we can't use even 1D slices (columns), I'd call this a patch not a slab, and it's better
                # to support extraction from the C++ side, rather then assemble it in Python.
                if self.slabSubsliceNdim == 0:
                    raise RuntimeError("For now we don't support slabs slicing the last axis.")

                self.slabIndex = [None] * ndim  # type: List[BufferedNdArray.Query.SlabAxisDesc]
                for dim, x in enumerate(queryNorm):
                    if x is None:
                        self.slabIndex[dim] = BufferedNdArray.Query.SlabAxisDesc(0, self._shape[dim], True)
                    elif isinstance(x, int):
                        self._check_index(x, dim)
                        self.slabIndex[dim] = BufferedNdArray.Query.SlabAxisDesc(x, x + 1, False)
                    elif isinstance(x, slice):
                        start = x.start if x.start is not None else 0
                        end = x.stop if x.stop is not None else self._shape[dim]

                        self.slabIndex[dim] = BufferedNdArray.Query.SlabAxisDesc(start, end, True)
                    else:
                        raise RuntimeError("Unexpected type {} in the query: {}".format(type(x), queryRaw))

                self.slabShape = tuple((i.end - i.start for i in self.slabIndex))

                return

            raise RuntimeError("Unsupported query format: {}".format(queryRaw))

        def _check_index(self, index: int, dim: int):
            if not 0 <= index < self._shape[dim]:
                raise IndexError("Index {} is out of range for axis {}.".format(index, dim))

            return True

    def __init__(self, filepath: str, fileMode: FileMode, shape: Tuple,
                 dtype: Union[np.dtype, Type], maxBufferSize: int = int(2e8), deleteOnDestruct: bool = False):
        assert(dtype == np.uint8 or dtype == np.float32)
        assert(isinstance(maxBufferSize, (int, float)))

        self.filepath = filepath
        self.fileMode = fileMode  # type: BufferedNdArray.FileMode
        self.dtype = np.dtype(dtype)  # type: np.dtype
        self.ndim = len(shape)
        self.shape = shape
        self.maxBufferSize = int(maxBufferSize)
        self.deleteOnDestruct = deleteOnDestruct

        # Take care of some smaller things in Python before calling the ctor.
        # If the parent folder doesn't exist, create it.
        if self.fileMode == BufferedNdArray.FileMode.rewrite and not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        self._cPointer = CppWrapper.bna_construct(self.filepath, self.fileMode.value, self.shape,
                                                  self.dtype, self.maxBufferSize)

        # Defines how to convert this instance when calling C++ function using ctypes.
        self._as_parameter_ = ctypes.c_void_p(self._cPointer)

        # Precompute slice sizes, which could be useful during indexes.
        # For each dimension it specifies the flat array size of a corresponding slice.
        self._sliceSizes = [1] * self.ndim
        for dim in range(self.ndim - 2, -1, -1):
            self._sliceSizes[dim] = self.shape[dim + 1] * self._sliceSizes[dim + 1]

    def destruct(self):
        CppWrapper.bna_destruct(self)
        if self.deleteOnDestruct:
            os.remove(self.filepath)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destruct()

    def __getitem__(self, queryRaw) -> Union[float, np.ndarray]:
        query = BufferedNdArray.Query(queryRaw, self.shape)

        if query.type == BufferedNdArray.Query.Type.cell:
            return CppWrapper.bna_read(self, query.cellIndex)
        elif query.type == BufferedNdArray.Query.Type.slice:
            output = np.empty(self.shape[-query.sliceNdim:], dtype=self.dtype)
            CppWrapper.bna_read_slice(self, query.sliceIndex, query.sliceNdim, output)
            return output
        elif query.type == BufferedNdArray.Query.Type.slab_outer:
            # C++ slab implementation currently supports only outer slabs.
            output = np.empty(query.slabShape, dtype=self.dtype)
            # indicesLow = np.zeros(self.ndim, np.uint64)
            indicesLow = np.asarray([x.start for x in query.slabIndex], dtype=np.uint64)
            indicesHigh = np.asarray([x.end for x in query.slabIndex], dtype=np.uint64)

            CppWrapper.bna_read_slab(self, indicesLow, indicesHigh, output)

            return output
        elif query.type == BufferedNdArray.Query.Type.slab:
            output = np.empty(query.slabShape, dtype=self.dtype)

            # todo this should probably be implemented in C++.
            slabStartIndex = [i.start for i in query.slabIndex]
            slabSubslicesNumber = query.slabShape[:-query.slabSubsliceNdim]
            slabSubslicesNumberFlat = self._multiply(slabSubslicesNumber)
            for subsliceIndexFlat in range(slabSubslicesNumberFlat):
                subsliceRelIndex = self._unflatten_index(slabSubslicesNumber, subsliceIndexFlat)
                subsliceIndex = tuple((slabStartIndex[dim] + subsliceRelIndex[dim]
                                       for dim in range(len(slabSubslicesNumber))))

                CppWrapper.bna_read_slice(self, subsliceIndex, query.slabSubsliceNdim, output[subsliceRelIndex])

            # Simplify the returned shape by removing cherry-picked axes (indexed by an int).
            slabShapeSimple = tuple((x for dim, x in enumerate(query.slabShape) if query.slabIndex[dim].keepAxis))

            return output.reshape(slabShapeSimple)
        elif query.type == BufferedNdArray.Query.Type.full:
            output = np.empty(self.shape, dtype=self.dtype)
            for i in range(self.shape[0]):
                CppWrapper.bna_read_slice(self, (i,), self.ndim - 1, output[i, ...])

            return output
        else:
            raise RuntimeError("Unsupported/unrecognized query type: {} from query {}".format(query.type, queryRaw))

    def __setitem__(self, queryRaw, value: Union[float, np.ndarray]):
        query = BufferedNdArray.Query(queryRaw, self.shape)

        # Handle single-cell writes.
        if query.type == BufferedNdArray.Query.Type.cell:
            CppWrapper.bna_write(self, query.cellIndex, value)
            return

        # For larger writes, first check the that input is a valid array.
        assert isinstance(value, np.ndarray)
        value = self._check_buffer_continuous(value)

        if query.type == BufferedNdArray.Query.Type.slice:
            sliceShape = self.shape[-query.sliceNdim:]
            if not self._are_shapes_compatible(value.shape, sliceShape):
                raise ValueError("Incompatible shape: {} Expected: {}".format(value.shape, sliceShape))

            CppWrapper.bna_write_slice(self, query.sliceIndex, query.sliceNdim, value)
        elif query.type == BufferedNdArray.Query.Type.slab:
            raise NotImplementedError("Slab writing is not yet supported")  # todo
        elif query.type == BufferedNdArray.Query.Type.full:
            if not self._are_shapes_compatible(value.shape, self.shape):
                raise ValueError("Incompatible shape: {} Expected: {}".format(value.shape, self.shape))

            CppWrapper.bna_write_full(self, value)
        elif query.type == BufferedNdArray.Query.Type.slab_outer:
            # todo should be implemented in C++.
            if not self._are_shapes_compatible(query.slabShape, value.shape):
                raise ValueError("Incompatible shape: {} Expected: {}".format(value.shape, query.slabShape))

            for i in range(query.slabIndex[0].start, query.slabIndex[0].end):
                CppWrapper.bna_write_slice(self, (i,), self.ndim - 1, value[i - query.slabIndex[0].start])
        else:
            raise RuntimeError("Unsupported/unrecognized query type: {} from query {}".format(query.type, queryRaw))

    def set_direct_mode(self, isDirectMode: bool):
        CppWrapper.bna_set_direct_mode(self, isDirectMode)

    def flush(self, flushOsBuffer: bool = False):
        CppWrapper.bna_flush(self, flushOsBuffer)

    def fill_box(self, value, cornerLow: Tuple, cornerHigh: Tuple):
        CppWrapper.bna_fill_box(self, value, cornerLow, cornerHigh)

    def max(self):
        currentMax = self[(0,) * self.ndim]
        for i in range(self.shape[0]):
            currentMax = max(currentMax, np.max(self[i]))

        return currentMax
    
    def min(self):
        currentMin = self[(0,) * self.ndim]
        for i in range(self.shape[0]):
            currentMin = min(currentMin, np.min(self[i]))

        return currentMin

    def compute_efficiency_and_reset(self):
        efficiency = CppWrapper.bna_compute_buffer_efficiency(self)
        CppWrapper.bna_reset_counters(self)

        return efficiency

    def _get_dtype_info(self):
        return np.iinfo(self.dtype) if np.issubdtype(self.dtype, np.integer) else np.finfo(self.dtype)

    @classmethod
    def _check_buffer_continuous(cls, buffer: np.ndarray) -> np.ndarray:
        """
        If a noncontinuous numpy array is given for reading,
        then create a continuous copy of the array and warn the user.
        """
        if not buffer.flags['C_CONTIGUOUS']:
            warnings.warn("The given array is not a continuous buffer. "
                          "This can lead to incorrect reads. Forcing a copy.",
                          category=BufferedNdArray.NoncontinuousArray)
            return buffer.copy()
        return buffer

    @classmethod
    def _unflatten_index(cls, shape: Tuple, indexFlat: int):
        """
        Convert a flat index into an N-d index (a tuple).
        """
        indexNd = tuple()
        for axis, length in enumerate(shape):
            sliceSize = np.prod(shape[axis + 1:], dtype=np.int64)
            axisIndex = int(indexFlat / sliceSize)
            indexNd += (axisIndex,)
            indexFlat -= axisIndex * sliceSize

        return indexNd

    @classmethod
    def _are_shapes_compatible(cls, shapeA: Tuple, shapeB: Tuple):
        """
        Check if shapes are equal, excluding any trailing 'one's.
        (Similar to NumPy broadcasting)
        """
        size = min(len(shapeA), len(shapeB))
        if shapeA[:size] != shapeB[:size]:
            return False

        return all((x == 1 for x in shapeA[size:])) and all((x == 1 for x in shapeB[size:]))

    @staticmethod
    def _multiply(iterable: Iterable):
        prod = 1
        for x in iterable:
            prod *= x
        return prod
