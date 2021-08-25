#pragma once

#ifdef __linux__
    #ifndef _LARGEFILE64_SOURCE
        #define _LARGEFILE64_SOURCE
    #endif
    #include <stdio.h>
#endif


#include <string>
#include <vector>
#include <fstream>
#include <functional>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cstddef>
#include <cstdio>

#include "PythonExtrasCLib.h"
#include "BufferedNdArray.hpp"
#include "PythonExtrasCLib.h"
#include "PythonExtrasCLib.h"
#include "macros.h"


// todo Dirty optimized code: need to declare a C-linkage function as friend.
extern "C" {
    __DLLEXPORT
    void aggregate_attention_volume(void* pAttentionRawArray, size_t* pDataSize, size_t* pPatchXSize,
                                    size_t* pPredictionStride, void* pAttentionOutArray);
}


template<typename TData>
class BufferedNdArray
{
public:
    // Though about using IndexNd here instead, but keeping the NDim dynamic
    // simplifies calling the wrapper functions from Python:
    // The void pointer to an array needs to be cast according to dtype and ndim,
    // which means we have to hard-code all combinations of dtype-ndim and decide
    // dynamically which function to call.
    // Methods using BNAs would have also needed to declare ndim explicitly,
    // which may not be desired in some cases.
    typedef std::vector<size_t> Tuple;

    enum class FileMode
    {
        Unknown = 0,
        Readonly = 1,
        Update = 2,
        Rewrite = 3
    };

    BufferedNdArray(std::wstring const& filepath,
                    FileMode mode,
                    Tuple const& shape,
                    size_t maxBufferSizeBytes);
    ~BufferedNdArray();
    
    BufferedNdArray(const BufferedNdArray& other) = delete;
    BufferedNdArray(BufferedNdArray&& other) = delete;
    BufferedNdArray& operator=(const BufferedNdArray& other) = delete;
    BufferedNdArray& operator=(BufferedNdArray&& other) = delete;


    void  Write(Tuple const& index, TData value);
    void  Write(size_t indexFlat, TData value);
    void WriteSlice(Tuple const& sliceIndex, size_t sliceNdim, TData const* pValues);
    void WritePatch(Tuple const& patchLowIndex, Tuple const& patchSize, TData const* pValues);
    void WriteFull(TData const* pValues);
    TData Read(Tuple const& index);
    TData Read(size_t indexFlat);
    void ReadSlice(Tuple const& sliceIndex, size_t sliceNdim, TData* pOutput);
    void ReadSlab(Tuple const& indicesLow, Tuple const& indicesHigh, TData* pOutput);
    void ReadRange(size_t indexStartFlat, size_t indexEndFlat, TData* pOutput);
    void FillBox(TData value, Tuple const& cornerLow, Tuple const& cornerHigh);

    void SetDirectMode(bool isDirectMode) { _isDirectMode = isDirectMode; }
    void FlushBuffer(bool flushOsBuffer = false);
    // ReSharper disable once CppMemberFunctionMayBeConst
    void CloseFile() { std::fclose(_file); }

    size_t GetSliceSizeFromNdim(size_t sliceNdim) const { return _sliceSizes[_ndim - sliceNdim - 1]; }
    size_t GetNdim() const { return _ndim; }
    Tuple GetShape() const { return _shape; }

    double ComputeBufferEfficiency() const;
    void ResetCounters();

    // todo Dirty optimized code: need to allow a function to access the private state. 
    friend void aggregate_attention_volume(void* pAttentionRawArray, size_t* pDataSize, size_t* pPatchXSize,
                                           size_t* pPredictionStride, void* pAttentionOutArray);

protected:
    /// Heuristic parameter, describes which index in the buffer is aligned to a missing element
    /// when the buffer is re-targeted.
    /// Value of 0 means that buffer only looks ahead, 0.5 means that the buffer is centered.
    const double _bufferShiftRatio = 0.25f;

    FileMode _fileMode;
    size_t _ndim;
    Tuple _shape;
    size_t _sizeFlat;
    Tuple _sliceSizes{};
    size_t _maxBufferSizeBytes;
    size_t _bufferSizeFlat{};
    size_t _bufferOffset{};
    bool _isBufferDirty{ false };
    bool _isDirectMode{ false };

    // Used for computing buffer efficiency.
    size_t _accessCounter{0};
    size_t _reloadCounter{0};

    Tuple _tempTuple{}; // Used for intermediate calculations to avoid allocating new tuples.

    std::vector<TData> _buffer{};
    FILE* _file{};

    void _retargetBuffer(size_t missingIndexFlat, bool withShift = true);
    size_t _assureIndexInBuffer(Tuple const& index);
    size_t _assureSliceInBuffer(Tuple const& sliceIndex, size_t sliceNdim);
    size_t _assureRangeInBuffer(size_t indexStartFlat, size_t indexEndFlat);

    void _throwIfRangeTooBig(size_t rangeSize) const;
    // std::fseek only supports only 32-bit offset values.
    static void _fseekPortable(FILE* file, uint64_t offset, int origin);
    size_t _ftellPortable(FILE* file) const;

    static std::string fileModeToString(FileMode mode);
};


//#include "BufferedNdArray.h"

template <typename TData>
BufferedNdArray<TData>::BufferedNdArray(std::wstring const& filepath, FileMode mode, 
    Tuple const& shape, size_t maxBufferSizeBytes)
    : _fileMode(mode), _ndim(shape.size()), _shape(shape), _maxBufferSizeBytes(maxBufferSizeBytes)
{
    auto multiplies = std::multiplies<>();  // todo check performance

    // Slice size for an axis is the total number of voxels selected when
    // slicing one element along the axis.
    // E.g. for an array[z, y, x], sliceSizes[0] is array.shape[1] * array.shape[2].
    _sliceSizes = Tuple(_ndim, size_t{ 1 });
    for (size_t dim = _ndim - 1; dim >= 1; dim--)
        _sliceSizes[dim - 1] = _shape[dim] * _sliceSizes[dim];

    _sizeFlat = std::accumulate(_shape.begin(), _shape.end(), size_t{ 1 }, multiplies);
    assert(_sizeFlat > 0);

    // Open the file. Truncate, since we're creating a new file.
    std::string filepathStr(filepath.begin(), filepath.end());
    _file = std::fopen(filepathStr.c_str(), fileModeToString(_fileMode).c_str());

    // To avoid hitting EOF when retargeting the buffer over a newly-created or extended file,
    // allocate the whole file before reading/writing.
    bool shouldInitFile = false;
    if (_fileMode != FileMode::Readonly)
    {
        _fseekPortable(_file, 0, SEEK_END);
        size_t position = _ftellPortable(_file);
        shouldInitFile = position < _sizeFlat * sizeof(TData) - 1;
    }
    
    if (shouldInitFile)
    {
        // Write the last byte of the file to force allocation.
        _fseekPortable(_file, _sizeFlat * sizeof(TData) - 1, SEEK_SET);
        std::fwrite("\0", sizeof(char), 1, _file);
    }

    // Create and position the buffer.
    _bufferOffset = 0;
    _bufferSizeFlat = std::min(maxBufferSizeBytes / sizeof(TData), _sizeFlat);

    _buffer = std::vector<TData>(_bufferSizeFlat);
    _retargetBuffer(_bufferOffset);

    // Init temp storage.
    _tempTuple.resize(_ndim);
}

template <typename TData>
BufferedNdArray<TData>::~BufferedNdArray()
{
    if (_isBufferDirty)
        FlushBuffer(true);

    std::fclose(_file);
}

template <typename TData>
void BufferedNdArray<TData>::Write(Tuple const& index, TData value)
{
    // Make sure that the target index is within the buffer.
    size_t relativeFlatIndex = _assureIndexInBuffer(index);

    // Mark buffer dirty, must be done after retargeting, which resets the flag.
    _isBufferDirty = true;
    _accessCounter++;

    _buffer[relativeFlatIndex] = value;
}

template <typename TData>
void BufferedNdArray<TData>::Write(size_t indexFlat, TData value)
{
    unflattenIndex_fast(indexFlat, _sliceSizes, _tempTuple);
    
    // Make sure that the target index is within the buffer.
    size_t relativeFlatIndex = _assureIndexInBuffer(_tempTuple);

    // Mark buffer dirty, must be done after retargeting, which resets the flag.
    _isBufferDirty = true;
    _accessCounter++;

    _buffer[relativeFlatIndex] = value;
}


template <typename TData>
void BufferedNdArray<TData>::WriteSlice(Tuple const& sliceIndex, size_t sliceNdim, TData const* pValues)
{
    size_t sliceSize = GetSliceSizeFromNdim(sliceNdim);
    _throwIfRangeTooBig(sliceSize); 

    // When in direct mode, write directly to the file.
    if (_isDirectMode)
    {
        FlushBuffer();  // Flush, if dirty.

        size_t indexFlat = flattenIndex_fast(sliceIndex, _sliceSizes);
        size_t sliceSize = _sliceSizes[_ndim - sliceNdim - 1];

        // Prepare to write to the file.
        _fseekPortable(_file, indexFlat * sizeof(TData), SEEK_SET);
        // Write the contents of the buffer to the file.
        std::fwrite(reinterpret_cast<void const*>(pValues), size_t{1}, sliceSize * sizeof(TData), _file);

        return;
    }

    size_t relFlatIndexStart = _assureSliceInBuffer(sliceIndex, sliceNdim);

    // Mark buffer dirty, must be done after retargeting, which resets the flag.
    _isBufferDirty = true;
    _accessCounter += sliceSize;

    std::copy(pValues, pValues + sliceSize, _buffer.begin() + relFlatIndexStart);

    // todo Some optimization can be made for large writes: either write straight to disk,
    // or avoid flushing/reloading the buffer, when it's not dirty and needed for write only.
}

template <typename TData>
void BufferedNdArray<TData>::WritePatch(Tuple const& patchLowIndex, Tuple const& patchSize, TData const* pValues)
{
    // We don't compute the exact patch flat size, just an upper bound which should be good enough.
    // Remember, that this isn't then number of voxels in the patch: it's the total range in the volume
    // that is required to be in-memory to read/write the patch.
    size_t approxPatchCoverage = (patchSize[0] + 1) * _sliceSizes[0];
    if (approxPatchCoverage >= static_cast<size_t>(_bufferSizeFlat * (1.0 - _bufferShiftRatio)))
    {
        // If a patch doesn't fit, we could read/write directly from/to the file.
        printf("Patch is bigger than buffer. Direct patch writes aren't implemented yet.\n");
        throw std::runtime_error("Patch is bigger than buffer. Direct patch writes aren't implemented yet.");
    }

    // We could be copying column-by-column here to make things faster,
    // but let's not over-optimize early.
    // We could call Write repeatedly, instead we check the range once and write directly to the buffer.
    size_t lowIndexFlat = flattenIndex_fast(patchLowIndex, _sliceSizes);
    _assureRangeInBuffer(lowIndexFlat,
                         std::min(lowIndexFlat + approxPatchCoverage, _sizeFlat));

    // todo try putting these calculations to the call site.
    size_t patchSizeFlat = std::accumulate(patchSize.begin(), patchSize.end(), 1, std::multiplies<>());
    Tuple patchSliceSizes = compute_slice_sizes(patchSize);

    // Mark buffer dirty, must be done after retargeting, which resets the flag.
    _isBufferDirty = true;
    _accessCounter += patchSizeFlat;
    
    Tuple dataIndexNd(_ndim, 0);
    Tuple innerIndexNd(_ndim, 0);
    for (size_t innerIndexFlat = 0; innerIndexFlat < patchSizeFlat; innerIndexFlat++)  // Loop over the patch.
    {
        unflattenIndex_fast(innerIndexFlat, patchSliceSizes, innerIndexNd);
        for (size_t dim = 0; dim < _ndim; dim++)
            dataIndexNd[dim] = patchLowIndex[dim] + innerIndexNd[dim];
        size_t dataIndexFlat = flattenIndex_fast(dataIndexNd, _sliceSizes);
        _buffer[dataIndexFlat - _bufferOffset] = *(pValues + innerIndexFlat);
    }
}

template <typename TData>
void BufferedNdArray<TData>::WriteFull(TData const* pValues)
{
    // If the whole dataset fits into the buffer, use it in a single write.
    if (_bufferSizeFlat >= _sizeFlat)
    {
        FlushBuffer();
        _bufferOffset = 0;  // Setting this manually, since _retargetBuffer() reloads from disk.
        std::copy(pValues, pValues + _sizeFlat, _buffer.data());
        _isBufferDirty = true;
        _accessCounter += _sizeFlat;
    }
    else
    {
        Tuple sliceIndex(_ndim, 0);
        for (size_t outerSliceIndex = 0; outerSliceIndex < _shape[0]; outerSliceIndex++)
        {
            sliceIndex[0] = outerSliceIndex;
            WriteSlice(sliceIndex, _ndim - 1, pValues + _sliceSizes[0] * outerSliceIndex);
        }
    }
}

template <typename TData>
TData BufferedNdArray<TData>::Read(Tuple const& index)
{
    _accessCounter++;

    // Make sure that the target index is within the buffer.
    size_t relativeFlatIndex = _assureIndexInBuffer(index);

    return _buffer[relativeFlatIndex];
}

template <typename TData>
TData BufferedNdArray<TData>::Read(size_t indexFlat)
{
    _accessCounter++;

    unflattenIndex_fast(indexFlat, _sliceSizes, _tempTuple);
    assert(indexFlat < _sizeFlat);

    // Make sure that the target index is within the buffer.
    size_t relativeFlatIndex = _assureIndexInBuffer(_tempTuple);

    return _buffer[relativeFlatIndex];
}


template <typename TData>
void BufferedNdArray<TData>::ReadSlice(Tuple const& sliceIndex, size_t sliceNdim, TData* pOutput)
{
    size_t sliceSize = GetSliceSizeFromNdim(sliceNdim);
    _throwIfRangeTooBig(sliceSize);

    _accessCounter += sliceSize;

    size_t relFlatIndexStart = _assureSliceInBuffer(sliceIndex, sliceNdim);

    std::copy(_buffer.data() + relFlatIndexStart,
              _buffer.data() + relFlatIndexStart + sliceSize,
              pOutput);
}

template <typename TData>
void BufferedNdArray<TData>::ReadSlab(Tuple const& indicesLow, Tuple const& indicesHigh, TData* pOutput)
{
    // todo For now we only support slabs of format (slice, None+). i.e. range on only the first axis.

    assert(indicesLow[0] != 0 || indicesHigh[0] != _shape[0]);  // The first axis should specify a range.

    for (size_t checkDim = 1; checkDim < _ndim; checkDim++)
    {
        assert(indicesLow[checkDim] == 0);                 // Remaining axes should index the whole axis.
        assert(indicesHigh[checkDim] == _shape[checkDim]); // Generic slabs aren't supported yet.
    }

    size_t sliceNdim = _ndim - 1;
    size_t sliceSize = GetSliceSizeFromNdim(sliceNdim);
    Tuple sliceIndex(sliceNdim, size_t{0});
    for (size_t index = indicesLow[0]; index < indicesHigh[0]; index++)
    {
        sliceIndex[0] = index;
        ReadSlice(sliceIndex, sliceNdim, pOutput + sliceSize * (index - indicesLow[0]));
    }
}

template <typename TData>
void BufferedNdArray<TData>::ReadRange(size_t indexStartFlat, size_t indexEndFlat, TData* pOutput)
{
    // This method assumes the range to be large, so it reads it directly from disk.

    assert(indexEndFlat <= _sizeFlat);

    if (_isBufferDirty)
        FlushBuffer();

    size_t elementsToRead = indexEndFlat - indexStartFlat;

    // Prepare to read from the file.
    _fseekPortable(_file, indexStartFlat * sizeof(TData), SEEK_SET);
    // Read the file contents into the buffer.
    // ReSharper disable once CppDeclaratorNeverUsed
    size_t bytesRead = std::fread(reinterpret_cast<void*>(pOutput), size_t{1},  elementsToRead * sizeof(TData), _file);

    assert(bytesRead == elementsToRead * sizeof(TData));
}


template <typename TData>
void BufferedNdArray<TData>::FillBox(TData value, Tuple const& cornerLow, Tuple const& cornerHigh)
{
    // First, flush the buffer to make sure that any current changes aren't lost.
    FlushBuffer();

    size_t lineSize = cornerHigh[_ndim - 1] - cornerLow[_ndim - 1];
    std::vector<TData> constantBuffer(lineSize, value);

    Tuple sourceIndex(cornerLow);
    while (true)
    {
        // Compute where in the file the line begins.
        size_t fileIndexFlat = 0;
        for (size_t dim = 0; dim < _ndim; dim++)
            fileIndexFlat += sourceIndex[dim] * _sliceSizes[dim];

        // Copy a line.
        _fseekPortable(_file, fileIndexFlat * sizeof(TData), SEEK_SET);
        std::fwrite(reinterpret_cast<void const*>(constantBuffer.data()), size_t{1}, lineSize * sizeof(TData), _file);

        // Increment the Nd index, doing something similar to an integer register.
        // Sweep right-to-left through the dims that are maxed out (full loop finished).
        // Use "dim+1" as the loop var to avoid the underflow of 'size_t'.
        // Do not consider the last dimension, since we copy data line-by-line along last dim.
        size_t dimPlusOne = _ndim - 1; // Ignore last dim.
        while (dimPlusOne > 0 && sourceIndex[dimPlusOne - 1] == cornerHigh[dimPlusOne - 1] - 1)
            dimPlusOne--;

        // All dims are at max, we're done.
        if (dimPlusOne == 0)
            break;

        // Increment the next non-maxed-out dimension.
        sourceIndex[dimPlusOne - 1]++;

        // Sweep back and reset the maxed out dimensions.
        for (dimPlusOne += 1; dimPlusOne <= _ndim - 1; dimPlusOne++) // Ignore last dim.
            sourceIndex[dimPlusOne - 1] = cornerLow[dimPlusOne - 1];
    }

    // Finally, force the buffer to be reloaded from disk.
    _retargetBuffer(_bufferOffset);
}

template <typename TData>
void BufferedNdArray<TData>::FlushBuffer(bool flushOsBuffer)
{
    if (!_isBufferDirty)
        return;

    // Prepare to write to the file.
    _fseekPortable(_file, _bufferOffset * sizeof(TData), SEEK_SET);
    // Write the contents of the buffer to the file.
    std::fwrite(reinterpret_cast<void const*>(_buffer.data()), size_t{1}, _bufferSizeFlat * sizeof(TData), _file);

    if (flushOsBuffer)
        std::fflush(_file);

    _isBufferDirty = false;
}

template <typename TData>
double BufferedNdArray<TData>::ComputeBufferEfficiency() const
{
    if (_reloadCounter == 0)
        return 1.0;

    return static_cast<double>(_accessCounter) / (_reloadCounter * _bufferSizeFlat);
}

template <typename TData>
void BufferedNdArray<TData>::ResetCounters()
{
    _accessCounter = 0;
    _reloadCounter = 0;
}

template <typename TData>
void BufferedNdArray<TData>::_retargetBuffer(size_t missingIndexFlat, bool withShift)
{
    // Calling code assumption: this function always reloads from disk (even if the missing index is in the buffer.)

    _reloadCounter++;

    // Update the buffer position.
    double shiftRatio = withShift ? _bufferShiftRatio : 0.0;
    size_t bufferShift = static_cast<size_t>(_bufferSizeFlat * shiftRatio);
    ptrdiff_t desiredOffset = static_cast<ptrdiff_t>(missingIndexFlat) - static_cast<ptrdiff_t>(bufferShift);
    ptrdiff_t maxAllowedOffset = static_cast<ptrdiff_t>(_sizeFlat) - static_cast<ptrdiff_t>(_bufferSizeFlat);
    _bufferOffset = std::max(ptrdiff_t{0}, std::min(desiredOffset, maxAllowedOffset));

    // Prepare to read from the file.
    _fseekPortable(_file, _bufferOffset * sizeof(TData), SEEK_SET);
    // Read the file contents into the buffer.
    size_t bytesRead = std::fread(reinterpret_cast<void*>(_buffer.data()), size_t{1}, _bufferSizeFlat * sizeof(TData), _file);
    
    assert(bytesRead == _bufferSizeFlat * sizeof(TData));
}

template <typename TData>
size_t BufferedNdArray<TData>::_assureIndexInBuffer(Tuple const& index)
{
    size_t indexFlat = flattenIndex_fast(index, _sliceSizes);
    ptrdiff_t relativeFlatIndex = indexFlat - _bufferOffset; // Can be negative.

    // Check if the index falls inside the buffer.
    if (relativeFlatIndex < 0 || static_cast<size_t>(relativeFlatIndex) >= _bufferSizeFlat)
    {
        FlushBuffer();
        _retargetBuffer(indexFlat);

        relativeFlatIndex = indexFlat - _bufferOffset;

        // Because we just adjusted the buffer.
        assert(relativeFlatIndex >= 0);
        assert(static_cast<size_t>(relativeFlatIndex) < _bufferSizeFlat);
    }

    return relativeFlatIndex;
}

template <typename TData>
size_t BufferedNdArray<TData>::_assureSliceInBuffer(Tuple const& sliceIndex, size_t sliceNdim)
{
    // Allow both exact indices, and with trailing zeros for sliced dimensions.
    // We do not support skipping over axes when slicing.
    assert(sliceIndex.size() >= _ndim - sliceNdim);

    size_t indexFlat = flattenIndex_fast(sliceIndex, _sliceSizes);
    size_t sliceSize = _sliceSizes[_ndim - sliceNdim - 1];

    return _assureRangeInBuffer(indexFlat, indexFlat + sliceSize);
}


template <typename TData>
size_t BufferedNdArray<TData>::_assureRangeInBuffer(size_t indexStartFlat, size_t indexEndFlat)
{
    ptrdiff_t rangeSize = indexEndFlat - indexStartFlat;
    assert(rangeSize > 0);
    // The interval is half-open.
    assert(indexEndFlat <= _sizeFlat);
    _throwIfRangeTooBig(rangeSize); // Don't just assert, check in Release builds as well.

    // Can be negative.
    ptrdiff_t relIndexStart = indexStartFlat - _bufferOffset;
    ptrdiff_t relIndexEnd = indexEndFlat - _bufferOffset;

    // Check if the range falls inside the buffer.
    if (relIndexStart < 0 || static_cast<size_t>(relIndexEnd) > _bufferSizeFlat)
    {
        FlushBuffer();
        _retargetBuffer(indexStartFlat, false);

        relIndexStart = indexStartFlat - _bufferOffset;

        // Because we just adjusted the buffer.
        assert(relIndexStart >= 0);
        assert(static_cast<size_t>(relIndexStart + rangeSize) <= _bufferSizeFlat);
    }

    return relIndexStart;
}

template <typename TData>
void BufferedNdArray<TData>::_throwIfRangeTooBig(size_t rangeSize) const
{
    // Used  to consider the shift too, but when ranges are loaded, no shift is used.
    // Removing the check helps with buffers equal in size to the whole dataset.
    // static_cast<size_t>(_bufferSizeFlat * (1.0 - _bufferShiftRatio))
    if (rangeSize > _bufferSizeFlat)
    {
        // If a slice doesn't fit, we could read/write directly from/to the file.
        printf("Requested range is bigger than buffer. Direct slice read/writes aren't implemented yet.\n");
        throw std::runtime_error("Slice is bigger than buffer. Direct slice read/writes aren't implemented yet.");
    }
}

template <typename TData>
void BufferedNdArray<TData>::_fseekPortable(FILE* file, uint64_t offset, int origin)
{
    // Make sure it's safe to cast to a signed int.
    assert(offset < static_cast<size_t>(std::numeric_limits<int64_t>::max()));

#ifdef _WIN32
    _fseeki64(file, static_cast<int64_t>(offset), origin);
#else
    #ifdef __linux__
        fseeko64(file, static_cast<int64_t>(offset), origin);
    #else
        #error "Only windows and linux platforms are supported."
    #endif
#endif
}

template <typename TData>
size_t BufferedNdArray<TData>::_ftellPortable(FILE* file) const
{
#ifdef _WIN32
    return static_cast<size_t>(_ftelli64(file));
#else
    #ifdef __linux__
        return static_cast<size_t>(ftello64(file));
    #else
        #error "Only windows and linux platforms are supported."
    #endif
#endif
}

template <typename TData>
std::string BufferedNdArray<TData>::fileModeToString(FileMode mode)
{
    switch (mode)
    {
    case FileMode::Readonly: return "rb";
    case FileMode::Update: return "rb+";
    case FileMode::Rewrite: return "wb+";
    default: throw std::runtime_error{std::string("Unknown file mode provided: ") + std::to_string(static_cast<int>(mode))};
    }
}

