#pragma once
#include <utility>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <functional>
#include <random>
#include <thread>
#include <atomic>

#include "PythonExtrasCLib.h"
#include "BufferedNdArray.hpp"


template<typename TData>
class MtPatchExtractor
{
protected:

    const size_t _ndim = 4;


    BufferedNdArray<TData>* _pVolumeData;

    // todo can we use Index4d here as well? Should be possible, replace all vectors.
    std::vector<size_t> _volumeSize;
    size_t _featureNumber;
    std::vector<size_t> _patchXSize;
    std::vector<size_t> _patchYSize;
    std::vector<size_t> _patchStride;
    std::vector<size_t> _patchInnerStride;
    size_t _predictionDelay;
    bool _detectEmptyPatches;
    TData _emptyValue;
    size_t _emptyCheckFeature;  // Which feature to use for the empty space check.

    float_t _undersamplingProbAny;
    float_t _undersamplingProbEmpty;
    float_t _undersamplingProbNonempty;


    size_t _inputBufferSize{};
    size_t _threadNumber;


    std::multiplies<size_t> _multiplies{ std::multiplies<size_t>() };  // NOLINT(modernize-use-transparent-functors) Breaks on Linux.
                                                                       // Prepare the random distribution for undersampling.
    std::random_device _randomDevice{};
    /// Create a random number generator for each thread.
    std::vector<std::default_random_engine> _randomEngines;

    std::vector<size_t> _patchNumber;
    Index4d _patchYOffset; // How far the Y-patch is from the X-patch. (lower corners)
    size_t _volumeSizeFlat;
    size_t _patchNumberFlat;
    size_t _patchXSizeFlat;
    size_t _patchYSizeFlat;

    std::vector<size_t> _patchXColumnNumber;
    std::vector<size_t> _patchYColumnNumber;
    size_t _patchXColumnNumberFlat;
    size_t _patchYColumnNumberFlat;

    size_t _xColumnSize;
    size_t _yColumnSize;
    Index4d _patchNumberSS;
    IndexNd<3> _patchXColumnNumberSS;
    IndexNd<3> _patchYColumnNumberSS;
    Index4d _volumeSizeSS;

    std::vector<TData> _inputDataBuffer;

public:

    //////// CONSTRUCTOR ////////

    MtPatchExtractor(BufferedNdArray<TData>* pVolumeData,
                     std::vector<size_t> const& volumeSize, // Doesn't include the feature dimension.
                     size_t featureNumber,
                     std::vector<size_t> const& patchXSize,
                     std::vector<size_t> const& patchYSize,
                     std::vector<size_t> const& patchStride,
                     std::vector<size_t> const& patchInnerStride,
                     size_t predictionDelay,
                     bool detectEmptyPatches,
                     TData emptyValue,
                     size_t emptyCheckFeature,

                     float_t undersamplingProbAny,
                     float_t undersamplingProbEmpty,
                     float_t undersamplingProbNonempty,

                     size_t inputBufferSize,
                     size_t threadNumber) 
        :
        _pVolumeData(pVolumeData),
        _volumeSize(volumeSize),
        _featureNumber(featureNumber),
        _patchXSize(patchXSize),
        _patchYSize(patchYSize),
        _patchStride(patchStride),
        _patchInnerStride(patchInnerStride),
        _predictionDelay(predictionDelay),
        _detectEmptyPatches(detectEmptyPatches),
        _emptyValue(emptyValue),
        _emptyCheckFeature(emptyCheckFeature),
        _undersamplingProbAny(undersamplingProbAny),
        _undersamplingProbEmpty(undersamplingProbEmpty),
        _undersamplingProbNonempty(undersamplingProbNonempty),
        _threadNumber(threadNumber)
    {
         
        if (_volumeSize.size() != _ndim || _patchXSize.size() != _ndim || _patchYSize.size() != _ndim || 
            _patchStride.size() != _ndim || _patchInnerStride.size() != _ndim)
            throw std::runtime_error("Invalid number of dimensions. Expected four.");

        if (_patchInnerStride[3] != 1)
        {
            // If you're implementing support for spatial strides, consider adjusting how the y-offset is computed.
            printf("Inner stride is probably broken, since we copy patches by columns. \n");
            throw std::runtime_error("Inner stride is probably broken, since we copy patches by columns. \n");
        }

        if (_emptyCheckFeature >= _featureNumber)
            throw std::runtime_error("Empty check feature index is out of range. \n");

        // Prepare the random generators for use in undersampling.
        for (size_t i = 0; i < _threadNumber; i++)
            _randomEngines.emplace_back(_randomDevice()); // Seed from a true random device.

        // Compute the offset of the Y-patch from the X-patch.
        _patchYOffset[0] = (_patchXSize[0] - 1) * _patchInnerStride[0] + _predictionDelay;
        for (size_t dim = 1; dim < _ndim; dim++)
            _patchYOffset[dim] = _patchXSize[dim] / 2 * _patchInnerStride[dim] - _patchYSize[dim] / 2;
        // ^ NB: better offset would be ((xsize-1)*stride - ysize)/2, 
        // which centers the y-patch relative to the x-patch support,
        // but this differs from the old behavior: y-value (center of y-patch now) should be at position
        // which is included in the X-patch. Keeping the old formula for better compatibility.
        // This shouldn't matter unless spatial stride is used. (which is currently not supported)

        _patchNumber = compute_patch_number(volumeSize, patchXSize, patchYSize, patchStride,
                                            patchInnerStride, predictionDelay);
        _patchNumberFlat = std::accumulate(_patchNumber.begin(), _patchNumber.end(), size_t{1}, _multiplies);

        _volumeSizeFlat = std::accumulate(_volumeSize.begin(), _volumeSize.end(), size_t{1}, _multiplies);
        _patchXSizeFlat = std::accumulate(_patchXSize.begin(), _patchXSize.end(), size_t{1}, _multiplies);
        _patchYSizeFlat = std::accumulate(_patchYSize.begin(), _patchYSize.end(), size_t{1}, _multiplies);

        // For efficiency, we don't copy data element-by-element, but copy continuous columns in the memory.
        // When dealing with columns, we simply ignore the last dimension (arrays are C-ordered)
        // in index computations and copy whole lines along that dimension.
        // todo measure efficiency again.

        _patchXColumnNumber = std::vector<size_t>(_patchXSize.begin(), _patchXSize.end() - 1);
        _patchYColumnNumber = std::vector<size_t>(_patchYSize.begin(), _patchYSize.end() - 1);

        _patchXColumnNumberFlat = std::accumulate(_patchXColumnNumber.begin(),
                                                  _patchXColumnNumber.end(), size_t{1}, _multiplies);
        _patchYColumnNumberFlat = std::accumulate(_patchYColumnNumber.begin(),
                                                  _patchYColumnNumber.end(), size_t{1}, _multiplies);

        // Length of a single column.
        _xColumnSize = _patchXSize[_ndim - 1];
        _yColumnSize = _patchYSize[_ndim - 1];

        // Optimization: precompute all the vectors that we'll need, instead of doing it in the loop.
        _patchNumberSS = compute_slice_sizes_fast<4>(_patchNumber);
        _volumeSizeSS  = compute_slice_sizes_fast<4>(_volumeSize);
        _patchXColumnNumberSS = compute_slice_sizes_fast<3>(_patchXColumnNumber);
        _patchYColumnNumberSS = compute_slice_sizes_fast<3>(_patchYColumnNumber);

        _inputBufferSize = std::min(inputBufferSize, _volumeSizeFlat * _featureNumber);
        _inputDataBuffer = std::vector<TData>(_inputBufferSize);
    }

    /////// BATCH CALL ////////

    void ExtractBatch(size_t batchStartIndex,
                      size_t batchSize,
                      TData* pOutX,
                      TData* pOutY,
                      size_t* pOutIndices,

                      size_t* pOutPatchesChecked,
                      size_t* pOutPatchesEmpty,
                      size_t* pOutPatchesExtracted,
                      size_t* pOutNextBatchIndex,
                      bool* pOutInputEndReached)
    {
        // Load a relevant chunk of the volume data into the input buffer.
        Index4d volumeIndexToLoad = _patchIndexToVolumeIndex(batchStartIndex);
        size_t bufferStartFlat = flattenIndex_fast(volumeIndexToLoad, _volumeSizeSS);
        // All math is done in 'cell indices', but remember, we have an extra 'features' dimension in the volume.
        size_t bufferEndFlat = std::min(bufferStartFlat + _inputBufferSize / _featureNumber, 
                                        _volumeSizeFlat);
        _pVolumeData->ReadRange(bufferStartFlat * _featureNumber, 
                                bufferEndFlat * _featureNumber,
                                _inputDataBuffer.data());

        std::vector<std::thread> threads{};
        std::atomic<size_t> globalBufferOffset{0};

        size_t patchesToProcess = std::min(batchSize, _patchNumberFlat - batchStartIndex);
        size_t chunkSize = patchesToProcess / _threadNumber;

        // Output buffers.
        std::vector<size_t> patchesChecked(_threadNumber, 0);
        std::vector<size_t> patchesEmpty(_threadNumber, 0);
        std::vector<size_t> patchesExtracted(_threadNumber, 0);
        std::vector<size_t> nextBatchIndex(_threadNumber, 0);
        bool* inputEndReached = new bool[_threadNumber];  // can't use std::vector<bool>, it packs them as bits.

        for (size_t threadIndex = 0; threadIndex < _threadNumber; threadIndex++)
        {
            // Each thread gets allocated a fixed chunk of the input.
            // But a thread can write out an arbitrary number of patches, due to undersampling,
            // running out of input buffer or empty patch skipping.
            size_t chunkOffset = threadIndex * chunkSize;
            size_t actualChunkSize = threadIndex < _threadNumber - 1 ? chunkSize : patchesToProcess - chunkOffset;

            threads.emplace_back([&, threadIndex, chunkOffset, actualChunkSize]() // Capture local vars by value!
            {
                _extractBatchThread(
                    _inputDataBuffer.data(),
                    bufferStartFlat, 
                    bufferEndFlat,

                    batchStartIndex + chunkOffset,
                    actualChunkSize,
                    globalBufferOffset,
                    _randomEngines[threadIndex],

                    pOutX, 
                    pOutY, 
                    pOutIndices,
                    &patchesChecked[threadIndex], 
                    &patchesEmpty[threadIndex], 
                    &patchesExtracted[threadIndex], 
                    &nextBatchIndex[threadIndex], 
                    &inputEndReached[threadIndex]);
            });
        }

        for (auto& thread : threads)
            thread.join();

        // Stop collecting results after encountering the first thread that couldn't finish.
        bool endReached = false;
        size_t totalPatchesChecked = 0;
        size_t totalPatchesEmpty = 0;
        size_t totalPatchesExtracted = 0;
        size_t lastNextBatchIndex = nextBatchIndex[_threadNumber - 1]; // By default, the last thread is the last ;).
        for (size_t i = 0; i < _threadNumber; i++)
        {
            // printf("Thread %zu extracted %zu patches and skipped %zu. Run out: %d\n", i, patchesExtracted[i], patchesEmpty[i], inputEndReached[i]);
            totalPatchesChecked   += patchesChecked[i];
            totalPatchesEmpty     += patchesEmpty[i];
            totalPatchesExtracted += patchesExtracted[i];
            if (!endReached && inputEndReached[i])
            {
                endReached = true;
                // If we didn't have enough data - start over (next batch) at the first thread that had to stop.
                lastNextBatchIndex = nextBatchIndex[i];
            }
            else if (endReached)
            {
                // Validate the assumption that if a thread runs out of input data,
                // then all the following threads extracted zero patches.
                // We assume that patches are laid out linearly wrt. to input,
                // if one thread requires an input element with at least index X, all following
                // threads require that or even higher indices.
                if (patchesExtracted[i] > 0)
                {
                    printf("ASSUMPTION FAILED: We have run out of input data, \n");
                    printf("but the next thread %zu still extracted %zu patches. ", i, patchesExtracted[i]);
                    abort();
                }
            }
        }

        delete[] inputEndReached;

        // Do a consistency check: number of global output buffer increments should be the same 
        // as the sum of local patch counters.
        if (totalPatchesExtracted != globalBufferOffset)
        {
            printf("FAILED THE CONSISTENCY CHECK: PATCHES MISSING DUE TO A RACE CONDITION?\n");
            printf("Expected %zu patches to be written, got %zu instead\n", 
                totalPatchesExtracted, globalBufferOffset.load());
            abort();
        }

        *pOutPatchesChecked = totalPatchesChecked;
        *pOutPatchesEmpty = totalPatchesEmpty;
        *pOutPatchesExtracted = totalPatchesExtracted;
        *pOutNextBatchIndex = lastNextBatchIndex;
        *pOutInputEndReached = endReached;
    }

protected:

    ////// THREAD ///////

    ///
    /// The function is made const because it is used by multiple threads.
    ///
    void _extractBatchThread(TData const* pVolumeData,
                             // These count volume cells, not values. Each cell has 'featureNumber' values in it.
                             size_t bufferStartFlat,
                             size_t bufferEndFlat,

                             size_t batchStartIndex,
                             size_t batchSize,
                             std::atomic<size_t>& globalBufferOffset,
                             std::default_random_engine& randomEngine,

                             TData* pOutX,
                             TData* pOutY,
                             size_t* pOutIndices,
                             size_t* pOutPatchesChecked,
                             size_t* pOutPatchesEmpty,
                             size_t* pOutPatchesExtracted,
                             size_t* pOutNextBatchIndex,
                             bool* pOutInputEndReached) const
    {
        // todo check and update all comments.
        // todo implement another two undersampling types.
        
        // Preallocate local variables, don't reallocate in a loop.
        // todo check if this is still necessary, they should be on the stack now.
        Index4d dataIndexNdX{};
        Index4d dataIndexNdY{};
        Index4d patchIndexNd{};

        // Allocate memory for storing a single patch that is being processed.
        // When it's assembled, it will be copied to the global buffer.
        // If we don't have an intermediate buffer, another thread can write over our results.
        // todo should this be pre-allocated in ctor?
        std::vector<TData> patchXData = std::vector<TData>(_patchXSizeFlat * _featureNumber);
        std::vector<TData> patchYData = std::vector<TData>(_patchYSizeFlat * _featureNumber);

        // Create a random distribution using a provided generator (engine).
        auto randomDist = std::uniform_real_distribution<float_t>(0.0f, 1.0f);

        // This function supports batching, i.e. we only extract 'batchSize' patches 
        // starting with 'batchStartIndex' patch.
        // Loop over all patches in a batch. Skip 'empty' patches.
        // We loop over a flat index and then unflatten it. We could write 'ndim' nested loops,
        // but this way is a little less verbose and more flexible.

        // Consistency check: A thread should be allocated at least some work.
        if (batchStartIndex >= _patchNumberFlat)
        {
            printf("Thread's start index is larger than the total number of patches.\n");
            throw std::runtime_error("Thread's start index is larger than the total number of patches.");
        }

        bool dontUndersampleAny      = _undersamplingProbAny > 0.999;
        bool dontUndersampleEmpty    = _undersamplingProbEmpty > 0.999;
        bool dontUndersampleNonempty = _undersamplingProbNonempty > 0.999;


        bool inputEndReached = false;
        size_t patchesChecked = 0;
        size_t patchesEmpty = 0;
        size_t patchesExtracted = 0;
        size_t patchIndexFlat = batchStartIndex;
        // Batch counts input patches, not the output (like in single-threaded code). 
        // I.e. each index names one of all the possible volume patches, which might
        // not be included in the end.
        // This makes the code more deterministic, i.e. we are sure that at the end
        // all input has been processed.
        // But this also means, that fewer (or even zero) patches could be returned.
        size_t batchEndIndex = batchStartIndex + batchSize;
        while (patchIndexFlat < batchEndIndex && patchIndexFlat < _patchNumberFlat) // NB: more break conditions below, due to convenience.
        {
            // Skip some of the patches outright according to the 'any' probability.
            if (dontUndersampleAny || randomDist(randomEngine) < _undersamplingProbAny)
            {
                // Compute the nd patch index. (Identify the patch)
                unflattenIndex_fast(patchIndexFlat, _patchNumberSS, patchIndexNd);

                // Figure out where in the orig. data the X-patch and Y-patch begin.
                for (size_t dim = 0; dim < _ndim; dim++)
                {
                    dataIndexNdX[dim] = patchIndexNd[dim] * _patchStride[dim];
                    dataIndexNdY[dim] = patchIndexNd[dim] * _patchStride[dim] + _patchYOffset[dim];
                }

                bool xIsEmpty;
                bool yIsEmpty;

                // Copy both the X-patch and the Y-patch into a temporary single-patch buffer.

                _copyPatch(pVolumeData, bufferStartFlat, bufferEndFlat,
                           dataIndexNdX,
                           _patchXColumnNumberSS, _patchXColumnNumberFlat, _xColumnSize,
                           patchXData, inputEndReached, xIsEmpty);
                _copyPatch(pVolumeData, bufferStartFlat, bufferEndFlat,
                           dataIndexNdY,
                           _patchYColumnNumberSS, _patchYColumnNumberFlat, _yColumnSize,
                           patchYData, inputEndReached, yIsEmpty);

                // Stop when there's not enough data in the input buffer to extract the next patch.
                if (inputEndReached)
                    break;

                // Perform empty/nonempty undersampling.
                bool isPatchEmpty = xIsEmpty && yIsEmpty;
                bool isPatchIncluded;
                if (isPatchEmpty)
                    isPatchIncluded = dontUndersampleEmpty    || randomDist(randomEngine) < _undersamplingProbEmpty;
                else
                    isPatchIncluded = dontUndersampleNonempty || randomDist(randomEngine) < _undersamplingProbNonempty;

                if (isPatchIncluded)
                {
                    // Claim output buffer space by advancing the atomic counter.
                    // Atomic fetch_add performs read-modify-write as a single operation, so we are thread safe.
                    size_t outputOffset = globalBufferOffset.fetch_add(1);

                    // Where in the output array should we write.
                    size_t outputOffsetXRaw = outputOffset * _patchXSizeFlat * _featureNumber;
                    size_t outputOffsetYRaw = outputOffset * _patchYSizeFlat * _featureNumber;
                    size_t outputOffsetIndices = outputOffset * _ndim;

                    // Write the results.
                    std::copy(patchXData.begin(), patchXData.end(), pOutX + outputOffsetXRaw);
                    std::copy(patchYData.begin(), patchYData.end(), pOutY + outputOffsetYRaw);
                    std::copy(dataIndexNdX.begin(), dataIndexNdX.end(), pOutIndices + outputOffsetIndices);

                    // Count how many patches this thread has extracted.
                    patchesExtracted += 1;
                }

                // Also, count the number of empty patches encountered.
                if (isPatchEmpty)
                    patchesEmpty += 1;

                patchesChecked += 1;
            }

            patchIndexFlat += 1;
        }

        // Return the information about the extracted batch.
        *pOutPatchesChecked   = patchesChecked;
        *pOutPatchesEmpty     = patchesEmpty;
        *pOutPatchesExtracted = patchesExtracted;
        *pOutNextBatchIndex   = patchIndexFlat;
        *pOutInputEndReached  = inputEndReached;
    }


    ///
    /// Copy a single patch over to a patch buffer. Could be an X-patch or a Y-patch.
    /// For performance reasons some of the local variables are pre-allocated as class members.
    /// The function is made const because it is used by multiple threads.
    ///
    void _copyPatch(TData const* pData,
                    size_t dataStartFlat,
                    size_t dataEndFlat,

                    Index4d const& dataIndexNd,
                    IndexNd<3> const& patchColumnNumberSS,
                    size_t patchColumnNumberFlat,
                    size_t columnSize,

                    std::vector<TData>& outPatchData,
                    bool& outInputEndReached, bool& outPatchIsEmpty) const
    {
        // Preallocate local variables, don't reallocate in a loop.
        // todo check if this is still necessary, they should be on the stack now.
        IndexNd<3> columnIndexNd{};
        Index4d sourceIndexNd{};

        bool isPatchEmpty = _detectEmptyPatches; // Init to false, if not skipping empty patches. (Don't skip anything).
        for (size_t columnIndexFlat = 0; columnIndexFlat < patchColumnNumberFlat; columnIndexFlat++)
        {
            // Compute the nd column index.
            unflattenIndex_fast(columnIndexFlat, patchColumnNumberSS, columnIndexNd);

            // Where the column starts in the original data .
            for (size_t dim = 0; dim < _ndim - 1; dim++)
                sourceIndexNd[dim] = dataIndexNd[dim] + columnIndexNd[dim] * _patchInnerStride[dim];

            // Handle the last 'column' dimension: point to its start, since we take all the data.
            sourceIndexNd[_ndim - 1] = dataIndexNd[_ndim - 1];

            size_t sourceIndexFlat = flattenIndex_fast(sourceIndexNd, _volumeSizeSS);

            if (sourceIndexFlat < dataStartFlat)
            {
                printf("FAIL small index %zu \n", sourceIndexFlat);
            }

            size_t sourceIndexRel = sourceIndexFlat - dataStartFlat;

            // The input data is buffered, i.e. we only have a chunk of it.
            // Check if the buffer has the data we need.
            if (sourceIndexFlat + columnSize > dataEndFlat)
            {
                outInputEndReached = true;
                return;
            }

            // todo Measure performance of the empty check. Maybe look into C++17 execution policies.
            size_t firstIndex = sourceIndexRel * _featureNumber;
            size_t lastIndex = (sourceIndexRel + columnSize) * _featureNumber;
            if (isPatchEmpty)  // Stop checking when we already know that the patch isn't empty.
            {
                // Check if the patch is empty. Check only one of the features.
                for (size_t i = firstIndex + _emptyCheckFeature; i < lastIndex + _emptyCheckFeature; i += _featureNumber)
                {
                    if (pData[i] != _emptyValue)
                    {
                        isPatchEmpty = false;
                        break;
                    }
                }
            }

            // Copy the whole column, even if it's empty. We don't know if the whole patch is empty or not.
            std::copy(&pData[firstIndex], &pData[lastIndex], 
                      outPatchData.data() + columnIndexFlat * columnSize * _featureNumber);
        }
        outPatchIsEmpty = isPatchEmpty;
    }

    ///
    /// Convert a flat patch index into a volume index pointing
    /// to the lower corner of that patch.
    ///
    Index4d _patchIndexToVolumeIndex(size_t patchIndexFlat)
    {
        Index4d patchIndexNd{};
        unflattenIndex_fast(patchIndexFlat, _patchNumberSS, patchIndexNd);

        Index4d volumeIndexNd{};
        for (size_t dim = 0; dim < _ndim; dim++)
            volumeIndexNd[dim] = patchIndexNd[dim] * _patchStride[dim];

        return volumeIndexNd;
    }
};
