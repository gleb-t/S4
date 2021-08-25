#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <vector>
#include <numeric>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <functional>
#include <random>
#include <thread>
#include <atomic>
#include <chrono>
#include <ctime>
#include <iostream>

#include "macros.h"
#include "BufferedNdArray.hpp"
#include "PythonExtrasCLib.h"


template<typename T>
void resize_array_point(void* pInputRaw, int sourceWidth, int sourceHeight, int sourceDepth,
						void* pOutputRaw, int targetWidth, int targetHeight, int targetDepth) 
{
	T* pInput = static_cast<T*>(pInputRaw);
	T* pOutput = static_cast<T*>(pOutputRaw);

	for (int x = 0; x < targetWidth; x++)
	{
		double tX = targetWidth > 1 ? static_cast<double>(x) / (targetWidth - 1) : 0;
		int sourceX = lround(tX * (sourceWidth - 1));

		for (int y = 0; y < targetHeight; y++)
		{
			double tY = targetHeight > 1 ? static_cast<double>(y) / (targetHeight - 1) : 0;
			int sourceY = lround(tY * (sourceHeight - 1));

			for (int z = 0; z < targetDepth; z++)
			{
				double tZ = targetDepth > 1 ? static_cast<double>(z) / (targetDepth - 1) : 0;
				int sourceZ = lround(tZ * (sourceDepth - 1));

				size_t sourceIndexFlat = sourceX * sourceHeight * sourceDepth + sourceY * sourceDepth + sourceZ;
				pOutput[x * targetHeight * targetDepth + y * targetDepth + z] = pInput[sourceIndexFlat];
			}
		}
	}
}


///
/// Compute the number of patches along each *patched* dimension.
/// Old function kept for compatibility. New ones don't use 'source axes' and patch all the dimensions.
///
std::vector<size_t> compute_patch_number_generic(const std::vector<size_t>& dataSize, const std::vector<size_t>& sourceAxes,
    const std::vector<size_t>& patchSize, const std::vector<size_t>& patchStride,
    const std::vector<size_t>& patchInnerStride, size_t predictionDelay)
{
    std::vector<size_t> patchNumber(sourceAxes.size());
    for (size_t i = 0; i < sourceAxes.size(); i++)
    {
        size_t dim = sourceAxes[i];
        size_t stride = patchStride[i];
        // How many voxels a patch covers.
        // Last point in time (Y-value) is 'predictionDelay' frames away from the previous frame.
        // E.g. if 'lastFrameGap' is 1, it immediately follows it.
        size_t patchSupport = i > 0 ? (patchSize[i] - 1) * patchInnerStride[i] + 1
            : (patchSize[i] - 2) * patchInnerStride[i] + 1 + predictionDelay;
        size_t totalPatchNumber = dataSize[dim] - patchSupport + 1;
        patchNumber[i] = (totalPatchNumber + stride - 1) / stride;  // Round up.
    }

    return patchNumber;
}


//todo this code (and functions called by it) has too many vector allocations. look at '4d_fast' methods for optimization.
// The problem mostly is that the std vector doesn't have a small size optimization,
// and always allocates stuff on the heap.
template<typename T>
void extract_patches_batched(void* pDataVoid, void* pOutputVoid, size_t* pOutputCenters,
                             size_t ndim, const std::vector<size_t>& dataSize,
                             const std::vector<size_t>& sourceAxes,
                             const std::vector<size_t>& patchSize, const std::vector<size_t>& patchStride,
                             size_t firstPatchIndex, size_t patchesPerBatch, bool isBatchSizedBuffer = false)
{
    T* pData = static_cast<T*>(pDataVoid);
    T* pOutput = static_cast<T*>(pOutputVoid);

    auto multiplies = std::multiplies<size_t>(); // Cache the functor, don't recreate it in a loop.

                                                 // Number of patched dimensions.
    size_t nPatchDim = sourceAxes.size();

    std::vector<size_t> patchInnerStride(nPatchDim, size_t{1});
    // Number of patches along the patched dimensions.
    std::vector<size_t> patchNumber = compute_patch_number_generic(dataSize, sourceAxes, patchSize, patchStride, 
                                                                   patchInnerStride, 1ULL);

    // Compute the (data)size of each patch. 
    // (Depends on the spatial extent a.k.a. 'patch size' of each patch, and on the size of the orig. data.
    std::vector<size_t> patchDataSize(ndim, 0);
    for (size_t dim = 0; dim < ndim; dim++)
    {
        auto itSourceDim = std::find(sourceAxes.begin(), sourceAxes.end(), dim);
        if (itSourceDim != sourceAxes.end())
        {
            // If the dimension is patched, use the patch size.
            size_t patchDim = itSourceDim - sourceAxes.begin();
            patchDataSize[dim] = patchSize[patchDim];
        }
        else
        {
            // Otherwise, take all data along that dimension.
            patchDataSize[dim] = dataSize[dim];
        }
    }
    // Total number of elements in a patch.
    size_t patchDataSizeFlat = std::accumulate(patchDataSize.begin(), patchDataSize.end(), 
                                               size_t{1}, multiplies);

    std::vector<size_t> patchCenterShift(nPatchDim);
    for (size_t patchDim = 0; patchDim < nPatchDim; patchDim++)
        patchCenterShift[patchDim] = patchSize[patchDim] / 2;

    // For efficiency, we don't copy data element-by-element, but copy
    // continuous columns in the memory.
    // When dealing with columns, we simply ignore the last dimension (arrays are C-ordered)
    // in index computations and copy whole lines along that dimension.

    // The number of columns in each dimension of the orig. data.
    std::vector<size_t> patchDataColumnNumber = std::vector<size_t>(patchDataSize.begin(), patchDataSize.end() - 1);
    size_t patchDataColumnNumberFlat = std::accumulate(patchDataColumnNumber.begin(), patchDataColumnNumber.end(), 
                                                       size_t{1}, multiplies);
    // Length of a single column.
    size_t columnSize = patchDataSize[ndim - 1];

    // This function supports batching, i.e. we only extract 'patchesPerBatch' patches 
    // starting with 'firstPatchIndex' patch.

    // Loop over all patches in a batch.
    // Since the number of dimensions is dynamic, we loop over a flat index
    // and then unflatten it.
    for (size_t indexFlat = firstPatchIndex; indexFlat < firstPatchIndex + patchesPerBatch; indexFlat++)
    {
        std::vector<size_t> patchIndexNd = unflattenIndex(indexFlat, patchNumber);

        // Figure out where in the orig. data the patch begins.
        // For patched dimensions, this is index * stride.
        // For the rest it's zero, since the whole dim. is copied.
        std::vector<size_t> dataSelectorStart(ndim, 0);
        std::vector<size_t> patchCenter(nPatchDim);
        for (size_t dim = 0; dim < ndim; dim++)
        {
            auto itSourceDim = std::find(sourceAxes.begin(), sourceAxes.end(), dim);
            if (itSourceDim != sourceAxes.end())
            {
                size_t patchDim = itSourceDim - sourceAxes.begin();
                dataSelectorStart[dim] = patchIndexNd[patchDim] * patchStride[patchDim];

                // Keep the location of the patch's center, which needs to be returned to the caller.
                patchCenter[patchDim] = dataSelectorStart[dim] + patchCenterShift[patchDim];
            }
        }

        // Where in the output array should we write.
        // All the patches are stacked one after another.
        size_t outputOffset = indexFlat * patchDataSizeFlat;
        size_t centerOutputOffset = indexFlat * nPatchDim;


        // If the output buffer is batch-sized. Adjust the offset to the batch.
        if (isBatchSizedBuffer)
        {
            outputOffset = (indexFlat - firstPatchIndex) * patchDataSizeFlat;
            centerOutputOffset = (indexFlat - firstPatchIndex) * nPatchDim;
        }

        for (size_t columnIndexFlat = 0; columnIndexFlat < patchDataColumnNumberFlat; columnIndexFlat++)
        {
            std::vector<size_t> columnIndexNd = unflattenIndex(columnIndexFlat, patchDataColumnNumber);

            // Where the column starts in the original data .
            std::vector<size_t> sourceIndexNd(ndim);
            for (size_t dim = 0; dim < ndim; dim++)
                sourceIndexNd[dim] = dataSelectorStart[dim] + columnIndexNd[dim];

            // Handle the last 'column' dimension: point to its start, we take all the data.
            sourceIndexNd[ndim - 1] = dataSelectorStart[ndim - 1];

            size_t sourceIndexFlat = flattenIndex(sourceIndexNd, dataSize);

            // Copy a whole column.
            std::copy(&pData[sourceIndexFlat], &pData[sourceIndexFlat + columnSize],
                      pOutput + outputOffset + columnIndexFlat * columnSize);
        }

        // Copy the patch center.
        std::copy(patchCenter.begin(), patchCenter.end(), pOutputCenters + centerOutputOffset);
    }
}

template<typename T>
void extract_patches(void* pDataVoid, void* pOutputVoid, size_t* pOutputCenters,
                     size_t ndim, const std::vector<size_t>& dataSize,
                     const std::vector<size_t>& sourceAxes,
                     const std::vector<size_t>& patchSize, const std::vector<size_t>& patchStride)
{
    auto multiplies = std::multiplies<size_t>(); // Cache the functor, don't recreate it in a loop.

    // Number of patches along the patched dimensions.
    std::vector<size_t> patchNumber = compute_patch_number_old(dataSize, sourceAxes, patchSize, patchStride, 1ULL);
    // Total flat number of patches that will be returned.
    size_t patchNumberFlat = std::accumulate(patchNumber.begin(), patchNumber.end(), size_t{1}, multiplies);

    extract_patches_batched<T>(pDataVoid, pOutputVoid, pOutputCenters, ndim, dataSize, sourceAxes,
                               patchSize, patchStride,
                               0, patchNumberFlat);
}


/**
 * \brief 
 * 
 * Extract patches/windows from a 4-dimensional array.
 * Each patch gets split into training data: X and Y.
 * X holds the whole hypercube, except for the last frame. Y holds a single scalar
 * from the center of the last frame. (Time is the first dimension, C-order is assumed.)
 * 'Empty' patches are those, where all values in X and the Y value are equal to the 'empty value'.
 * Empty patches do not get extracted.
 * Extraction is performed in batches, returning control after 'batchSize' patches were extracted.
 * 
 */
template<typename T>
void extract_patched_training_data_without_empty_4d(
                T* pData, 
                size_t dataStartFlat, size_t dataEndFlat,
                const std::vector<size_t>& dataSize,
                const std::vector<size_t>& patchSize,
                const std::vector<size_t>& patchStride,
                const std::vector<size_t>& patchInnerStride,
                size_t lastFrameGap,
                bool skipEmptyPatches, T emptyValue,
                size_t batchStartIndex, size_t batchSize,
                float_t undersamplingProb,
    
                T* pOutX, T* pOutY, size_t* pOutIndices,
                size_t* pOutPatchesExtracted, size_t* pOutNextBatchIndex, bool* pOutInputEndReached)
{
    // Cache the functor, don't recreate it in a loop.
    auto multiplies = std::multiplies<size_t>();
    // Prepare the random distribution for undersampling.
    std::random_device r;
    std::default_random_engine randomEngine(r());
    std::uniform_real_distribution<float_t> randomDist(0.0f, 1.0f);

    const size_t ndim = 4;

    if (dataSize.size() != ndim || patchSize.size() != ndim || patchStride.size() != ndim)
        throw std::runtime_error("Invalid number of dimensions. Expected four.");

    if (patchInnerStride[3] != 1)
    {
        printf("Inner stride is probably broken, since we copy patch by columns. \n");
        throw std::runtime_error("Inner stride is probably broken, since we copy patch by columns. \n");
    }

    // Number of patches along each dimension.
    std::vector<size_t> patchNumber = compute_patch_number_old(dataSize, patchSize, patchStride,
                                                               patchInnerStride, lastFrameGap);
    // Total flat number of patches.
    size_t patchNumberFlat = std::accumulate(patchNumber.begin(), patchNumber.end(), size_t{1}, multiplies);

    // Total number of elements in an 'X' patch.
    std::vector<size_t> patchSizeX(patchSize);
    // The 'X' part includes all timesteps but the last. The last timestep is used for 'Y'.
    patchSizeX[0] -= 1;
    size_t patchSizeXFlat = std::accumulate(patchSizeX.begin(), patchSizeX.end(), size_t{1}, multiplies);

    // For efficiency, we don't copy data element-by-element, but copy
    // continuous columns in the memory.
    // When dealing with columns, we simply ignore the last dimension (arrays are C-ordered)
    // in index computations and copy whole lines along that dimension.

    // The number of columns in each dimension.
    std::vector<size_t> patchXColumnNumber = std::vector<size_t>(patchSizeX.begin(), patchSizeX.end() - 1);
    size_t patchXColumnNumberFlat = std::accumulate(patchXColumnNumber.begin(),
                                                    patchXColumnNumber.end(), size_t{1}, multiplies);
    // Length of a single column.
    size_t columnSize = patchSize[ndim - 1];

    // This function supports batching, i.e. we only extract 'batchSize' patches 
    // starting with 'batchStartIndex' patch.
    // Loop over all patches in a batch. Skip 'empty' patches.
    // We loop over a flat index and then unflatten it. We could write 'ndim' nested loops,
    // but this way is a little less verbose and more flexible.

    // Optimization: prepare allocate all vectors that we'll need, instead doing it in the loop.
    Index4d patchNumberSS = compute_slice_sizes_fast<4>(patchNumber);
    IndexNd<3> patchXColumnNumberSS = compute_slice_sizes_fast<3>(patchXColumnNumber);
    Index4d dataSizeSS = compute_slice_sizes_fast<4>(dataSize);
    Index4d dataIndexNd{};
    Index4d patchIndexNd{};
    IndexNd<3> columnIndexNd{};
    Index4d sourceIndexNd{};
    Index4d sourceIndexNdY{};

    bool pInputEndReached = false;
    size_t patchesExtracted = 0;
    size_t indexFlat = batchStartIndex;
    while (patchesExtracted < batchSize && indexFlat < patchNumberFlat)
    {
        // Skip some of the patches according to the provided probability.
        float_t random = randomDist(randomEngine);
        bool dontUndersample = undersamplingProb > 0.999; // Floating-point comparison.
        if (dontUndersample || random < undersamplingProb)
        {
            unflattenIndex_fast(indexFlat, patchNumberSS, patchIndexNd);

            // Figure out where in the orig. data the patch begins.
            for (size_t dim = 0; dim < ndim; dim++)
                dataIndexNd.X[dim] = patchIndexNd.X[dim] * patchStride[dim];

            // Where in the output array should we write.
            // All the patches are stacked one after another.
            size_t outputOffsetX = patchesExtracted * patchSizeXFlat;
            size_t outputOffsetY = patchesExtracted;
            size_t outputOffsetIndices = patchesExtracted * ndim;

            bool xIsEmpty = skipEmptyPatches; // Init to false, if not skipping empty patches.
            for (size_t columnIndexFlat = 0; columnIndexFlat < patchXColumnNumberFlat; columnIndexFlat++)
            {
                unflattenIndex_fast(columnIndexFlat, patchXColumnNumberSS, columnIndexNd);

                // Where the column starts in the original data .
                for (size_t dim = 0; dim < ndim - 1; dim++)
                    sourceIndexNd.X[dim] = dataIndexNd.X[dim] + columnIndexNd.X[dim] * patchInnerStride[dim];

                // Handle the last 'column' dimension: point to its start, we take all the data.
                sourceIndexNd.X[ndim - 1] = dataIndexNd.X[ndim - 1];

                size_t sourceIndexFlat = flattenIndex_fast(sourceIndexNd, dataSizeSS);
                size_t sourceIndexRel = sourceIndexFlat - dataStartFlat;

                // The input data is buffered, i.e. we only have a chunk of it.
                // Check if the buffer has the data we need.
                if (sourceIndexFlat + columnSize >= dataEndFlat)
                {
                    pInputEndReached = true;
                    break;
                }
                

                // Check if the column is empty.
                auto first = &pData[sourceIndexRel];
                auto last = &pData[sourceIndexRel + columnSize];
                bool allValuesEqual = skipEmptyPatches && std::adjacent_find(first, last, std::not_equal_to<T>()) == last;

                xIsEmpty = xIsEmpty && *first == emptyValue && allValuesEqual;

                // Copy the whole column, even if it's empty. We don't know if the whole patch is empty or not.
                std::copy(first, last, pOutX + outputOffsetX + columnIndexFlat * columnSize);
            }

            // Extract Y.
            // Take the last timestep. Note that Y ignores the inner stride, and uses 'lastFrameGap' instead.
            sourceIndexNdY.X[0] = dataIndexNd.X[0] + (patchSize[0] - 2) * patchInnerStride[0] + lastFrameGap;
            for (size_t dim = 1; dim < ndim; dim++)
            {
                // Take the value in the middle of the patch.
                sourceIndexNdY.X[dim] = dataIndexNd.X[dim] + patchSize[dim] / 2 * patchInnerStride[dim];
            }

            size_t sourceIndexYFlat = flattenIndex_fast(sourceIndexNdY, dataSizeSS);
            size_t sourceIndexYRel = sourceIndexYFlat - dataStartFlat;

            // Check if the buffer has the data we need.
            if (pInputEndReached || sourceIndexYFlat >= dataEndFlat)
            {
                pInputEndReached = true;
                break;
            }

            T y = pData[sourceIndexYRel];
            bool yIsEmpty = y == emptyValue;

            if (!xIsEmpty || !yIsEmpty)
            {
                // Copy the results.
                *(pOutY + outputOffsetY) = y;
                std::copy(&dataIndexNd.X[0], &dataIndexNd.X[4], pOutIndices + outputOffsetIndices);

                // Advance the output offset.
                patchesExtracted += 1;
            }
        }

        indexFlat += 1;
    }

    // Return the information about the extracted batch.
    *pOutPatchesExtracted = patchesExtracted;
    *pOutNextBatchIndex = indexFlat;
    *pOutInputEndReached = pInputEndReached;
}


///
/// A multithreaded version of the same method. Uses an atomic counter to synchronize output to the buffer.
///
template<typename T>
void extract_patched_training_data_without_empty_4d_multi(
                            T* pData,
                            size_t dataStartFlat, size_t dataEndFlat,
                            const std::vector<size_t>& dataSize,
                            const std::vector<size_t>& patchSize,
                            const std::vector<size_t>& patchStride,
                            const std::vector<size_t>& patchInnerStride,
                            size_t lastFrameGap,
                            bool skipEmptyPatches, T emptyValue,
                            size_t batchStartIndex, size_t batchSize,
                            float_t undersamplingProb,

                            std::atomic<size_t>& globalBufferOffset,

                            T* pOutX, T* pOutY, size_t* pOutIndices,
                            size_t* pOutPatchesExtracted, size_t* pOutPatchesEmpty,
                            size_t* pOutNextBatchIndex, bool* pOutInputEndReached)
{
    // Cache the functor, don't recreate it in a loop.
    auto multiplies = std::multiplies<size_t>();
    // Prepare the random distribution for undersampling.
    std::random_device r;
    std::default_random_engine randomEngine(r());
    std::uniform_real_distribution<float_t> randomDist(0.0f, 1.0f);

    const size_t ndim = 4;

    if (dataSize.size() != ndim || patchSize.size() != ndim || patchStride.size() != ndim)
        throw std::runtime_error("Invalid number of dimensions. Expected four.");

    if (patchInnerStride[3] != 1)
    {
        printf("Inner stride is probably broken, since we copy patch by columns. \n");
        throw std::runtime_error("Inner stride is probably broken, since we copy patch by columns. \n");
    }

    // Number of patches along each dimension.
    std::vector<size_t> patchNumber = compute_patch_number_old(dataSize, patchSize, patchStride,
                                                               patchInnerStride, lastFrameGap);
    // Total flat number of patches.
    size_t patchNumberFlat = std::accumulate(patchNumber.begin(), patchNumber.end(), size_t{1}, multiplies);

    // Total number of elements in an 'X' patch.
    std::vector<size_t> patchSizeX(patchSize);
    // The 'X' part includes all timesteps but the last. The last timestep is used for 'Y'.
    patchSizeX[0] -= 1;
    size_t patchSizeXFlat = std::accumulate(patchSizeX.begin(), patchSizeX.end(), size_t{1}, multiplies);

    // For efficiency, we don't copy data element-by-element, but copy continuous columns in the memory.
    // When dealing with columns, we simply ignore the last dimension (arrays are C-ordered)
    // in index computations and copy whole lines along that dimension.

    // The number of columns in each dimension.
    std::vector<size_t> patchXColumnNumber = std::vector<size_t>(patchSizeX.begin(), patchSizeX.end() - 1);
    size_t patchXColumnNumberFlat = std::accumulate(patchXColumnNumber.begin(),
                                                    patchXColumnNumber.end(), size_t{1}, multiplies);
    // Length of a single column.
    size_t columnSize = patchSize[ndim - 1];

    // Consistency check: A thread should be allocated at least some work.
    if (batchStartIndex >= patchNumberFlat)
    {
        printf("Thread's start index is larger than the total number of patches.\n");
        throw std::runtime_error("Thread's start index is larger than the total number of patches.");
    }

    // This function supports batching, i.e. we only extract 'batchSize' patches 
    // starting with 'batchStartIndex' patch.
    // Loop over all patches in a batch. Skip 'empty' patches.
    // We loop over a flat index and then unflatten it. We could write 'ndim' nested loops,
    // but this way is a little less verbose and more flexible.

    // Optimization: allocate all the vectors that we'll need, instead of doing it in the loop.
    Index4d patchNumberSS = compute_slice_sizes_fast<4>(patchNumber);
    IndexNd<3> patchXColumnNumberSS = compute_slice_sizes_fast<3>(patchXColumnNumber);
    Index4d dataSizeSS = compute_slice_sizes_fast<4>(dataSize);
    Index4d dataIndexNd{};
    Index4d patchIndexNd{};
    IndexNd<3> columnIndexNd{};
    Index4d sourceIndexNd{};
    Index4d sourceIndexNdY{};

    // Allocate memory for storing a single patch that is being processed.
    // When it's assembled, it will be copied to the global buffer.
    // If we don't have an intermediate buffer, another thread can write over our results.
    std::vector<T> patchDataX(patchSizeXFlat);

    bool inputEndReached = false;
    size_t patchesExtracted = 0;
    size_t patchesEmpty = 0;
    size_t indexFlat = batchStartIndex;
    // Batch counts input patches, not the output (like in single-threaded code). 
    // This makes the code more deterministic, i.e. we are sure that at the end
    // all input has been processed.
    // But this also means, that fewer (or even zero) patches could be returned.
    size_t batchEndIndex = batchStartIndex + batchSize;
    while (indexFlat < batchEndIndex && indexFlat < patchNumberFlat) // Note: break conditions below, due to convenience.
    {
        // Skip some of the patches according to the provided probability.
        float_t random = randomDist(randomEngine);
        bool dontUndersample = undersamplingProb > 0.999; // Floating-point comparison.
        if (dontUndersample || random < undersamplingProb)
        {
            unflattenIndex_fast(indexFlat, patchNumberSS, patchIndexNd);

            // Figure out where in the orig. data the patch begins.
            for (size_t dim = 0; dim < ndim; dim++)
                dataIndexNd.X[dim] = patchIndexNd.X[dim] * patchStride[dim];

            bool xIsEmpty = skipEmptyPatches; // Init to false, if not skipping empty patches.
            for (size_t columnIndexFlat = 0; columnIndexFlat < patchXColumnNumberFlat; columnIndexFlat++)
            {
                unflattenIndex_fast(columnIndexFlat, patchXColumnNumberSS, columnIndexNd);

                // Where the column starts in the original data .
                for (size_t dim = 0; dim < ndim - 1; dim++)
                    sourceIndexNd.X[dim] = dataIndexNd.X[dim] + columnIndexNd.X[dim] * patchInnerStride[dim];

                // Handle the last 'column' dimension: point to its start, we take all the data.
                sourceIndexNd.X[ndim - 1] = dataIndexNd.X[ndim - 1];

                size_t sourceIndexFlat = flattenIndex_fast(sourceIndexNd, dataSizeSS);
                size_t sourceIndexRel = sourceIndexFlat - dataStartFlat;

                // The input data is buffered, i.e. we only have a chunk of it.
                // Check if the buffer has the data we need.
                if (sourceIndexFlat + columnSize >= dataEndFlat)
                {
                    inputEndReached = true;
                    break;
                }


                // Check if the column is empty.
                auto first = &pData[sourceIndexRel];
                auto last = &pData[sourceIndexRel + columnSize];
                bool allValuesEqual = skipEmptyPatches && std::adjacent_find(first, last, std::not_equal_to<T>()) == last;

                xIsEmpty = xIsEmpty && *first == emptyValue && allValuesEqual;

                // Copy the whole column, even if it's empty. We don't know if the whole patch is empty or not.
                std::copy(first, last, patchDataX.data() + columnIndexFlat * columnSize);
            }

            // Extract Y.
            // Take the last timestep. Note that Y ignores the inner stride, and uses 'lastFrameGap' instead.
            sourceIndexNdY.X[0] = dataIndexNd.X[0] + (patchSize[0] - 2) * patchInnerStride[0] + lastFrameGap;
            for (size_t dim = 1; dim < ndim; dim++)
            {
                // Take the value in the middle of the patch.
                sourceIndexNdY.X[dim] = dataIndexNd.X[dim] + patchSize[dim] / 2 * patchInnerStride[dim];
            }

            size_t sourceIndexYFlat = flattenIndex_fast(sourceIndexNdY, dataSizeSS);
            size_t sourceIndexYRel = sourceIndexYFlat - dataStartFlat;

            // Check if the buffer has the data we need.
            if (inputEndReached || sourceIndexYFlat >= dataEndFlat)
            {
                inputEndReached = true;
                break;
            }

            T y = pData[sourceIndexYRel];
            bool yIsEmpty = y == emptyValue;

            if (!xIsEmpty || !yIsEmpty)
            {
                // Claim output buffer space by advancing the atomic counter.
                // Atomic fetch_add performs read-modify-write as a single operation, so we are thread safe.
                size_t outputOffset = globalBufferOffset.fetch_add(1);

                // Where in the output array should we write.
                size_t outputOffsetX = outputOffset * patchSizeXFlat;
                size_t outputOffsetY = outputOffset;
                size_t outputOffsetIndices = outputOffset * ndim;

                // Write the results.
                std::copy(patchDataX.begin(), patchDataX.end(), pOutX + outputOffsetX);
                *(pOutY + outputOffsetY) = y;
                std::copy(&dataIndexNd.X[0], &dataIndexNd.X[4], pOutIndices + outputOffsetIndices);

                // Count how many patches this thread has extracted.
                patchesExtracted += 1;
            }
            else
            {
                patchesEmpty += 1;
            }
        }

        indexFlat += 1;
    }

    // Return the information about the extracted batch.
    *pOutPatchesExtracted = patchesExtracted;
    *pOutPatchesEmpty = patchesEmpty;
    *pOutNextBatchIndex = indexFlat;
    *pOutInputEndReached = inputEndReached;
}


/**
* 
* This version doesn't use undersampling, empty patch skipping or striding.
* It's meant for dense multi-threaded patch extraction.
* 
*/
template<typename T>
void extract_patched_training_data_dense_4d(T* pData,
                                            size_t dataStartFlat, size_t dataEndFlat,
                                            const std::vector<size_t>& dataSize,
                                            const std::vector<size_t>& patchSize,
                                            const std::vector<size_t>& patchInnerStride,
                                            size_t lastFrameGap,
                                            size_t batchStartIndex, size_t batchSize,

                                            T* pOutX, T* pOutY,
                                            size_t* pOutPatchesExtracted, 
                                            bool* pOutInputEndReached)
{
    const size_t ndim = 4;

    const std::vector<size_t> patchStride{ 1, 1, 1, 1 };

    // Cache the functor, don't recreate it in a loop.
    auto multiplies = std::multiplies<size_t>();

    if (dataSize.size() != ndim || patchSize.size() != ndim)
        throw std::runtime_error("Invalid number of dimensions. Expected four.");

    if (patchInnerStride[3] != 1)
    {
        printf("Inner stride is probably broken, since we copy patch by columns. \n");
        throw std::runtime_error("Inner stride is probably broken, since we copy patch by columns. \n");
    }

    // Number of patches along each dimension.
    std::vector<size_t> patchNumber = compute_patch_number_old(dataSize, patchSize, patchStride,
                                                               patchInnerStride, lastFrameGap);
    // Total flat number of patches.
    size_t patchNumberFlat = std::accumulate(patchNumber.begin(), patchNumber.end(), size_t{1}, multiplies);

    // Total number of elements in an 'X' patch.
    std::vector<size_t> patchSizeX(patchSize);
    // The 'X' part includes all timesteps but the last. The last timestep is used for 'Y'.
    patchSizeX[0] -= 1;
    size_t patchSizeXFlat = std::accumulate(patchSizeX.begin(), patchSizeX.end(), size_t{1}, multiplies);

    // For efficiency, we don't copy data element-by-element, but copy
    // continuous columns in the memory.
    // When dealing with columns, we simply ignore the last dimension (arrays are C-ordered)
    // in index computations and copy whole lines along that dimension.

    // The number of columns in each dimension.
    std::vector<size_t> patchXColumnNumber = std::vector<size_t>(patchSizeX.begin(), patchSizeX.end() - 1);
    size_t patchXColumnNumberFlat = std::accumulate(patchXColumnNumber.begin(),
                                                    patchXColumnNumber.end(), size_t{1}, multiplies);
    // Length of a single column.
    size_t columnSize = patchSize[ndim - 1];

    // This function supports batching, i.e. we only extract 'batchSize' patches 
    // starting with 'batchStartIndex' patch.
    // We loop over a flat index and then unflatten it. We could write 'ndim' nested loops,
    // but this way is a little less verbose and more flexible.

    // Optimization: allocate all vectors that we'll need, instead of doing it in the loop.
    Index4d patchNumberSS = compute_slice_sizes_fast<4>(patchNumber);
    Index4d dataSizeSS    = compute_slice_sizes_fast<4>(dataSize);
    IndexNd<3> patchXColumnNumberSS = compute_slice_sizes_fast<3>(patchXColumnNumber);

//    index4d_t dataIndexNd{};  <-- Is the same as patch index, since we have no striding.
    Index4d patchIndexNd{};
    IndexNd<3> columnIndexNd{};
    Index4d sourceIndexNd{};
    Index4d sourceIndexNdY{};

    bool inputEndReached = false;
    size_t patchesExtracted = 0;
    while (patchesExtracted < batchSize && batchStartIndex + patchesExtracted < patchNumberFlat)
    {
        // Since we don't skip patches, flat index follows 'patchesExtracted'.
        size_t indexFlat = batchStartIndex + patchesExtracted;
        unflattenIndex_fast(indexFlat, patchNumberSS, patchIndexNd);

        // Where in the output array should we write.
        // All the patches are stacked one after another.
        size_t outputOffsetX = patchesExtracted * patchSizeXFlat;
        size_t outputOffsetY = patchesExtracted;

        for (size_t columnIndexFlat = 0; columnIndexFlat < patchXColumnNumberFlat; columnIndexFlat++)
        {
            unflattenIndex_fast(columnIndexFlat, patchXColumnNumberSS, columnIndexNd);

            // Where the column starts in the original data .
            for (size_t dim = 0; dim < ndim - 1; dim++)
                sourceIndexNd.X[dim] = patchIndexNd.X[dim] + columnIndexNd.X[dim] * patchInnerStride[dim];

            // Handle the last 'column' dimension: point to its start, we take all the data.
            sourceIndexNd.X[ndim - 1] = patchIndexNd.X[ndim - 1];

            size_t sourceIndexFlat = flattenIndex_fast(sourceIndexNd, dataSizeSS);
            size_t sourceIndexRel = sourceIndexFlat - dataStartFlat;

            // The input data is buffered, i.e. we only have a chunk of it.
            // Check if the buffer has the data we need.
            if (sourceIndexFlat + columnSize >= dataEndFlat)
            {
                inputEndReached = true;
                break;
            }


            // Copy the whole column.
            auto first = &pData[sourceIndexRel];
            auto last = &pData[sourceIndexRel + columnSize];
            std::copy(first, last, pOutX + outputOffsetX + columnIndexFlat * columnSize);
        }

        // Extract Y.
        // Take the last timestep. Note that Y ignores the inner stride, and uses 'lastFrameGap' instead.
        sourceIndexNdY.X[0] = patchIndexNd.X[0] + (patchSize[0] - 2) * patchInnerStride[0] + lastFrameGap;
        for (size_t dim = 1; dim < ndim; dim++)
        {
            // Take the value in the middle of the patch.
            sourceIndexNdY.X[dim] = patchIndexNd.X[dim] + patchSize[dim] / 2 * patchInnerStride[dim];
        }

        size_t sourceIndexYFlat = flattenIndex_fast(sourceIndexNdY, dataSizeSS);
        size_t sourceIndexYRel = sourceIndexYFlat - dataStartFlat;

        // Check if the buffer has the data we need.
        if (inputEndReached || sourceIndexYFlat >= dataEndFlat)
        {
            inputEndReached = true;
            break;
        }

        // Copy the Y
        *(pOutY + outputOffsetY) = pData[sourceIndexYRel];

        // Advance the output offset.
        patchesExtracted += 1;
    }

    // Return the information about the extracted batch.
    *pOutPatchesExtracted = patchesExtracted;
    *pOutInputEndReached = inputEndReached;
}


template <typename T>
void sparse_insert_into_bna(BufferedNdArray<T>* pArray, size_t const* pIndices, T const* pValues,
                            size_t valueNumber)
{
    size_t ndim = pArray->GetNdim();
    typename BufferedNdArray<T>::Tuple indexNd(ndim);
    for (size_t i = 0; i < valueNumber; i++)
    {
        size_t const* pIndex = pIndices + i * ndim;
        std::copy(pIndex, pIndex + ndim, indexNd.data());
        pArray->Write(indexNd, *(pValues + i));
    }
}

template <typename T>
void sparse_insert_slices_into_bna(BufferedNdArray<T>* pArray, size_t const* pIndices, T const* pValues,
                                   size_t sliceNdim, size_t sliceNumber)
{
    size_t ndim = pArray->GetNdim();  // Total array axis number.
    size_t sliceIndexNdim = ndim - sliceNdim;  // Length of a slice index (non-sliced axis number).
    typename BufferedNdArray<T>::Tuple sliceIndexNd(sliceIndexNdim);
    size_t sliceSizeFlat = pArray->GetSliceSizeFromNdim(sliceNdim);
    for (size_t i = 0; i < sliceNumber; i++)
    {
        size_t const* pIndex = pIndices + i * sliceIndexNdim;
        std::copy(pIndex, pIndex + sliceIndexNdim, sliceIndexNd.data());
        pArray->WriteSlice(sliceIndexNd, sliceNdim, pValues + i * sliceSizeFlat);
    }
}

///
/// Insert patches at location specified by the indices (lower patch corner).
/// If 'isConstPatch' is false, expect a buffer with N patches, otherwise take a buffer with a single patch.
///
template <typename T>
void sparse_insert_patches_into_bna(BufferedNdArray<T>* pArray, size_t const* pIndices, T const* pValues,
                                    size_t const* pPatchSize, size_t patchNumber, bool isConstPatch)
{
    size_t ndim = pArray->GetNdim();
    typename BufferedNdArray<T>::Tuple patchSize(pPatchSize, pPatchSize + ndim);
    size_t patchSizeFlat = std::accumulate(patchSize.begin(), patchSize.end(), 1, std::multiplies<>());

    typename BufferedNdArray<T>::Tuple patchIndexNd(ndim, 0);
    for (size_t i = 0; i < patchNumber; i++)
    {
        size_t const* pIndex = pIndices + i * ndim;
        std::copy(pIndex, pIndex + ndim, patchIndexNd.data());
        if (!isConstPatch)
            pArray->WritePatch(patchIndexNd, patchSize, pValues + i * patchSizeFlat);
        else
            pArray->WritePatch(patchIndexNd, patchSize, pValues);  // Always write the same patch.
    }
}


template <typename T>
void sparse_insert_const_into_bna(BufferedNdArray<T>* pArray, size_t const* pIndices, T const& constValue,
                                  size_t valuesToInsert)
{
    size_t ndim = pArray->GetNdim();
    typename BufferedNdArray<T>::Tuple indexNd(ndim);

    for (size_t i = 0; i < valuesToInsert; i++)
    {
        size_t const* pIndex = pIndices + i * ndim;
        std::copy(pIndex, pIndex + ndim, indexNd.data());
        pArray->Write(indexNd, constValue);
    }
}

void _multithreading_test_worker(uint8_t* pData, uint64_t offset, uint64_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        size_t computationNumber = 20;
        size_t dummyResult = 0;
        for (size_t j = 0; j < computationNumber; j++)
        {
            dummyResult += 13 + dummyResult * 5 % 3;
        }
        pData[offset + i] += static_cast<uint8_t>(dummyResult % 256);
    }
}

//todo Keeping state between calls experiment.
uint64_t StaticState = 5;

extern "C" {

    // todo Get rid of the void pointers. Ctypes can handle typed pointers.


    // todo remove the test code.
	__DLLEXPORT
	void test(void* pInput, int width, int height) {
		double* pData = static_cast<double *>(pInput);
		for (int i = 0; i < width * height; ++i) {
			pData[i] = pData[i] * 2;
		}
	}

    __DLLEXPORT
    void static_state_test(uint64_t increment, uint64_t* out)
	{
        StaticState += increment;
        *out = StaticState;
	}

    __DLLEXPORT
    void multithreading_test(uint8_t* pData, uint64_t size, uint64_t threadNumber)
    {
        std::vector<std::thread> threads{};
        size_t chunkSize = size / threadNumber;
        for (size_t i = 0; i < threadNumber; i++)
        {
            size_t chunkOffset = i * chunkSize;
            size_t actualChunkSize = std::min(chunkSize, size - chunkOffset);
            threads.emplace_back([=]() { _multithreading_test_worker(pData, chunkOffset, actualChunkSize); });
        }

        for (auto& thread : threads)
            thread.join();
    }

    __DLLEXPORT
        void resize_array_point_float32(void* pInputRaw, int sourceWidth, int sourceHeight, int sourceDepth,
                                        void* pOutputRaw, int targetWidth, int targetHeight, int targetDepth) {
        resize_array_point<float_t>(pInputRaw, sourceWidth, sourceHeight, sourceDepth,
                                   pOutputRaw, targetWidth, targetHeight, targetDepth);

    }
    __DLLEXPORT
        void resize_array_point_float64(void* pInputRaw, int sourceWidth, int sourceHeight, int sourceDepth,
                                        void* pOutputRaw, int targetWidth, int targetHeight, int targetDepth) {
        resize_array_point<double_t>(pInputRaw, sourceWidth, sourceHeight, sourceDepth,
                                     pOutputRaw, targetWidth, targetHeight, targetDepth);

    }
    __DLLEXPORT
	void resize_array_point_uint8(void* pInputRaw, int sourceWidth, int sourceHeight, int sourceDepth,
								  void* pOutputRaw, int targetWidth, int targetHeight, int targetDepth) {
		resize_array_point<uint8_t>(pInputRaw, sourceWidth, sourceHeight, sourceDepth,
                                          pOutputRaw, targetWidth, targetHeight, targetDepth);

	}

    __DLLEXPORT
	void extract_patches_uint8(void* data, void* output, size_t* outputCenters, size_t ndim,
                               size_t* dataSize, size_t dataSizeL, 
                               size_t* sourceAxes, size_t sourceAxesL,
                               size_t* patchSize, size_t patchSizeL, 
                               size_t* patchStride, size_t patchStrideL)
	{
        extract_patches<uint8_t>(data, output, outputCenters, ndim,
                                       std::vector<size_t>(dataSize, dataSize + dataSizeL),
                                       std::vector<size_t>(sourceAxes, sourceAxes + sourceAxesL),
                                       std::vector<size_t>(patchSize, patchSize + patchSizeL),
                                       std::vector<size_t>(patchStride, patchStride + patchStrideL));
	}

    __DLLEXPORT
        void extract_patches_batched_uint8(void* data, void* output, size_t* outputCenters, size_t ndim,
                                           size_t* dataSize, size_t dataSizeL,
                                           size_t* sourceAxes, size_t sourceAxesL,
                                           size_t* patchSize, size_t patchSizeL,
                                           size_t* patchStride, size_t patchStrideL,
                                           size_t firstPatchIndex, size_t patchesPerBatch, bool isBatchSizedBuffer)
    {
        extract_patches_batched<uint8_t>(data, output, outputCenters, ndim,
                                               std::vector<size_t>(dataSize, dataSize + dataSizeL),
                                               std::vector<size_t>(sourceAxes, sourceAxes + sourceAxesL),
                                               std::vector<size_t>(patchSize, patchSize + patchSizeL),
                                               std::vector<size_t>(patchStride, patchStride + patchStrideL),
                                               firstPatchIndex, patchesPerBatch, isBatchSizedBuffer);
    }


    __DLLEXPORT
    void extract_patched_training_data_without_empty_4d_uint8(
        uint8_t* pData, 
        size_t dataStartFlat, 
        size_t dataEndFlat,
        size_t* dataSize, 
        size_t* patchSize, 
        size_t* patchStride, 
        size_t* patchInnerStride,
        size_t lastFrameGap,
        bool skipEmptyPatches, 
        uint8_t emptyValue,
        size_t batchStartIndex, 
        size_t batchSize,
        float_t undersamplingProb,
        uint8_t* pOutX, 
        uint8_t* pOutY, 
        size_t* pOutIndices,
        size_t* pOutPatchesExtracted, 
        size_t* pOutNextBatchIndex, 
        bool* pOutInputEndReached)
    {
        extract_patched_training_data_without_empty_4d<uint8_t>(pData, 
                                                                dataStartFlat, dataEndFlat,
                                                                std::vector<size_t>(dataSize, dataSize + 4),
                                                                std::vector<size_t>(patchSize, patchSize + 4),
                                                                std::vector<size_t>(patchStride, patchStride + 4),
                                                                std::vector<size_t>(patchInnerStride, patchInnerStride + 4),
                                                                lastFrameGap,
                                                                skipEmptyPatches, emptyValue,
                                                                batchStartIndex, batchSize, 
                                                                undersamplingProb,

                                                                pOutX, pOutY, pOutIndices,
                                                                pOutPatchesExtracted, pOutNextBatchIndex,
                                                                pOutInputEndReached);
    }

    __DLLEXPORT
    void extract_patched_training_data_without_empty_4d_multithreaded_uint8(
        uint8_t* pData,
        size_t dataStartFlat,
        size_t dataEndFlat,
        size_t* pDataSize,
        size_t* pPatchSize,
        size_t* pPatchStride,
        size_t* pPatchInnerStride,
        size_t lastFrameGap,
        bool skipEmptyPatches,
        uint8_t emptyValue,
        size_t batchStartIndex,
        size_t batchSize,
        float_t undersamplingProb,
        size_t threadNumber,

        uint8_t* pOutX,
        uint8_t* pOutY,
        size_t* pOutIndices,
        size_t* pOutPatchesExtracted,
        size_t* pOutNextBatchIndex,
        bool* pOutInputEndReached)
    {
        const size_t ndim = 4;

        std::vector<size_t> dataSize(pDataSize, pDataSize + ndim);
        std::vector<size_t> patchSize(pPatchSize, pPatchSize + ndim);
        std::vector<size_t> patchStride(pPatchStride, pPatchStride + ndim);
        std::vector<size_t> patchInnerStride(pPatchInnerStride, pPatchInnerStride + ndim);

        std::vector<std::thread> threads{};
        std::atomic<size_t> globalBufferOffset{0};

        std::vector<size_t> patchNumber = compute_patch_number_old(dataSize, patchSize, patchStride, patchInnerStride, lastFrameGap);
        // Total flat number of patches.
        size_t patchNumberFlat = std::accumulate(patchNumber.begin(), patchNumber.end(), size_t{1}, std::multiplies<>());

        size_t patchesToProcess = std::min(batchSize, patchNumberFlat - batchStartIndex);
        size_t chunkSize = patchesToProcess / threadNumber;

        // Output buffers.
        std::vector<size_t> patchesExtracted(threadNumber, 0);
        std::vector<size_t> patchesEmpty(threadNumber, 0);
        std::vector<size_t> nextBatchIndex(threadNumber, 0);
        bool* inputEndReached = new bool[threadNumber];

        for (size_t i = 0; i < threadNumber; i++)
        {
            // Each thread gets allocated a fixed chunk of the input.
            // But a thread can write out an arbitrary number of patches, due to undersampling,
            // running out of input buffer or empty patch skipping.
            size_t chunkOffset = i * chunkSize;
            size_t actualChunkSize = i < threadNumber - 1 ? chunkSize : patchesToProcess - chunkOffset;

            threads.emplace_back([&, i, chunkOffset, actualChunkSize]() // Capture local vars by value!
            {
                extract_patched_training_data_without_empty_4d_multi<uint8_t>(
                    pData,
                    dataStartFlat, dataEndFlat,
                    dataSize, patchSize, patchStride, patchInnerStride,
                    lastFrameGap,
                    skipEmptyPatches, emptyValue,

                    batchStartIndex + chunkOffset,
                    actualChunkSize,
                    undersamplingProb,

                    globalBufferOffset,

                    pOutX, pOutY, pOutIndices,
                    &patchesExtracted[i], &patchesEmpty[i], 
                    &nextBatchIndex[i], &inputEndReached[i]);
            });
        }

        for (auto& thread : threads)
            thread.join();

        // Stop collecting results after encountering the first thread that couldn't finish.
        bool endReached = false;
        size_t totalPatchesExtracted = 0;
        size_t lastNextBatchIndex = nextBatchIndex[threadNumber - 1]; // By default, the last thread is the last ;).
        for (size_t i = 0; i < threadNumber; i++)
        {
//            printf("Thread %zu extracted %zu patches and skipped %zu. Run out: %d\n", i, patchesExtracted[i], patchesEmpty[i], inputEndReached[i]);
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
                // We assume that patches are layed out linearly wrt. to input,
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

        *pOutPatchesExtracted = totalPatchesExtracted;
        *pOutNextBatchIndex = lastNextBatchIndex;
        *pOutInputEndReached = endReached;
    }
    
    __DLLEXPORT
    void extract_patched_training_data_multithreaded_uint8(
        uint8_t* pData, 
        size_t dataStartFlat, 
        size_t dataEndFlat,
        size_t* dataSize, 
        size_t* patchSize, 
        size_t* patchInnerStride,
        size_t lastFrameGap,
        size_t batchStartIndex, 
        size_t batchSize,
        size_t threadNumber,
        uint8_t* pOutX, 
        uint8_t* pOutY, 
        size_t* pOutPatchesExtracted, 
        size_t* pOutNextBatchIndex, 
        bool* pOutInputEndReached)
    {
        const int ndim = 4;

	    auto multiplies = std::multiplies<size_t>();
        // Total number of elements in an 'X' patch.
        std::vector<size_t> patchSizeX(patchSize, patchSize + 4);
        // The 'X' part includes all timesteps but the last. The last timestep is used for 'Y'.
        patchSizeX[0] -= 1;
        size_t patchSizeXFlat = std::accumulate(patchSizeX.begin(), patchSizeX.end(), size_t{1}, multiplies);


        std::vector<std::thread> threads{};
        size_t chunkSize = batchSize / threadNumber;

        // Output buffers.
        std::vector<size_t> patchesExtracted(threadNumber, 0);
        std::vector<size_t> nextBatchIndex(threadNumber, 0);
        bool* inputEndReached = new bool[threadNumber];

        for (size_t i = 0; i < threadNumber; i++)
        {
            size_t chunkOffset = i * chunkSize;
            size_t actualChunkSize = i < threadNumber - 1 ? chunkSize : batchSize - chunkOffset;
            threads.emplace_back([&, i, chunkOffset, actualChunkSize]()
            {
                extract_patched_training_data_dense_4d<uint8_t>(
                    pData,
                    dataStartFlat, dataEndFlat,
                    std::vector<size_t>(dataSize, dataSize + ndim),
                    std::vector<size_t>(patchSize, patchSize + ndim),
                    std::vector<size_t>(patchInnerStride, patchInnerStride + ndim),
                    lastFrameGap,
                    batchStartIndex + chunkOffset,
                    actualChunkSize,

                    pOutX + chunkOffset * patchSizeXFlat, 
                    pOutY + chunkOffset, 
                    &patchesExtracted[i], 
                    &inputEndReached[i]
                );
            });
        }

        for (auto& thread : threads)
            thread.join();

        // Stop collecting results after encountering the first thread that couldn't finish.
        bool endReached = false;
        size_t totalPatchesExtracted = 0;
        for (size_t i = 0; i < threadNumber; i++)
        {
            totalPatchesExtracted += patchesExtracted[i];
            if (inputEndReached[i])
            {
                endReached = true;
                break;
            }

        }

        *pOutPatchesExtracted = totalPatchesExtracted;
        *pOutNextBatchIndex = batchStartIndex + totalPatchesExtracted; // No empty skipping, so it's the same.
        *pOutInputEndReached = endReached;


        delete[] inputEndReached;
    }

    __DLLEXPORT
    void sparse_insert_into_bna_uint8(void* pArrayRaw,
                                      size_t const* pIndices,
                                      uint8_t const* pValues,
                                      size_t valuesToInsert)
    {
        BufferedNdArray<uint8_t>* pArray = reinterpret_cast<BufferedNdArray<uint8_t>*>(pArrayRaw);
        sparse_insert_into_bna<uint8_t>(pArray, pIndices, pValues, valuesToInsert);
    }

    __DLLEXPORT
    void sparse_insert_slices_into_bna_float32(void* pArrayRaw,
                                               size_t const* pIndices,
                                               float_t const* pValues,
                                               size_t sliceNdim,
                                               size_t valueNumber)
    {
        auto pArray = reinterpret_cast<BufferedNdArray<float_t>*>(pArrayRaw);
        sparse_insert_slices_into_bna<float_t>(pArray, pIndices, pValues, sliceNdim, valueNumber);
    }

    __DLLEXPORT
    void sparse_insert_patches_into_bna_uint8(void* pArrayRaw,
                                              size_t const* pIndices,
                                              uint8_t const* pValues,
                                              size_t const* pPatchSize,
                                              size_t patchNumber,
                                              bool isConstPatch)
	{
        BufferedNdArray<uint8_t>* pArray = reinterpret_cast<BufferedNdArray<uint8_t>*>(pArrayRaw);
        sparse_insert_patches_into_bna(pArray, pIndices, pValues, pPatchSize, patchNumber, isConstPatch);
	}
    __DLLEXPORT
    void sparse_insert_patches_into_bna_float32(void* pArrayRaw,
                                                size_t const* pIndices,
                                                float_t const* pValues,
                                                size_t const* pPatchSize,
                                                size_t patchNumber,
                                                bool isConstPatch)
	{
        BufferedNdArray<float_t>* pArray = reinterpret_cast<BufferedNdArray<float_t>*>(pArrayRaw);
        sparse_insert_patches_into_bna(pArray, pIndices, pValues, pPatchSize, patchNumber, isConstPatch);
	}

    __DLLEXPORT
    void sparse_insert_const_into_bna_uint8(void* pArrayRaw,
                                            size_t const* pIndices,
                                            uint8_t constValue,
                                            size_t valueNumber)
    {
        BufferedNdArray<uint8_t>* pArray = reinterpret_cast<BufferedNdArray<uint8_t>*>(pArrayRaw);
        sparse_insert_const_into_bna<uint8_t>(pArray, pIndices, constValue, valueNumber);
    }

    __DLLEXPORT
    void smooth_3d_array_average_float(float_t const* pInputData,
                                       size_t const* pDataSize,
                                       size_t kernelRadius,
                                       float_t* pOutputData)
    {
	    smooth_3d_array_average(pInputData, IndexNd<3>(pDataSize, pDataSize + 3), kernelRadius, pOutputData);
	}


    void upscale_attention_patch(float_t const* pAttPatchSource,
                                 std::vector<size_t> const& attPatchSize,
                                 std::vector<size_t> const& attPatchSliceSizes,
                                 std::vector<size_t> const& targetSize,
                                 Index4d const& targetSliceSizes,
                                 std::vector<float_t>& outputPatch)
	{
        for (size_t targetT = 0; targetT < targetSize[0]; targetT++)
        {
            size_t patchT = int(roundf(static_cast<float>(targetT) / (targetSize[0] - 1) * (attPatchSize[0] - 1)));
            for (size_t targetZ = 0; targetZ < targetSize[1]; targetZ++)
            {
                // An edge-case for 2D data.
                size_t patchZ = targetSize[1] > 1 ? 
                    int(roundf(static_cast<float>(targetZ) / (targetSize[1] - 1) * (attPatchSize[1] - 1)))
                    : 0;
                for (size_t targetY = 0; targetY < targetSize[2]; targetY++)
                {
                    size_t patchY = int(roundf(static_cast<float>(targetY) / (targetSize[2] - 1) * (attPatchSize[2] - 1)));
                    for (size_t targetX = 0; targetX < targetSize[3]; targetX++)
                    {
                        size_t patchX = int(roundf(static_cast<float>(targetX) / (targetSize[3] - 1) * (attPatchSize[3] - 1)));
                        size_t outputIndexFlat = targetT * targetSliceSizes.X[0] + targetZ * targetSliceSizes.X[1] +
                                                 targetY * targetSliceSizes.X[2] + targetX * targetSliceSizes.X[3];
                        size_t attIndexFlat = patchT * attPatchSliceSizes[0] + patchZ * attPatchSliceSizes[1] +
                                              patchY * attPatchSliceSizes[2] + patchX * attPatchSliceSizes[3];

                        outputPatch[outputIndexFlat] = *(pAttPatchSource + attIndexFlat);
                    }
                }
            }
        }
	}


    ///
    /// Aggregates a raw 8D attention volume into a 4D volume
    /// by adding attention from each patch to spatial positions.
    /// Essentially computes "overall voxel importance".
    ///
    // todo Move to a separate project?
    __DLLEXPORT
    void aggregate_attention_volume(void* pAttentionRawArray,
                                    size_t* pDataSize,
                                    size_t* pPatchXSize,
                                    size_t* pPredictionStride,
                                    void* pAttentionOutArray)
	{
        // todo prediction delay isn't needed anymore, because attention is written based on X-indices.

        constexpr size_t DataNdim = 4;
        constexpr size_t AttNdim = 8;

        auto pAttentionRaw = reinterpret_cast<BufferedNdArray<float_t>*>(pAttentionRawArray);
        auto pAttentionOut = reinterpret_cast<BufferedNdArray<float_t>*>(pAttentionOutArray);

	    std::vector<size_t> dataSize{ pDataSize, pDataSize + DataNdim };
	    std::vector<size_t> patchXSize{ pPatchXSize, pPatchXSize + DataNdim };
        Index4d patchXSizeNd{pPatchXSize, pPatchXSize + DataNdim};
	    std::vector<size_t> predictionStride{ pPredictionStride, pPredictionStride + DataNdim };
        std::vector<size_t> attVolSize = pAttentionRaw->GetShape();
        // Domain size includes only the spatiotemporal dimensions.
	    std::vector<size_t> attVolDomainSize{ attVolSize.data(), attVolSize.data() + DataNdim };

        std::vector<size_t> attPatchSize{ attVolSize.begin() + DataNdim, attVolSize.end() };

	    auto multiplesFunc = std::multiplies<>();
	    size_t attPatchSizeFlat = std::accumulate(attPatchSize.begin(), attPatchSize.end(), size_t{1}, multiplesFunc);
	    size_t patchXSizeFlat = std::accumulate(patchXSize.begin(), patchXSize.end(), size_t{1}, multiplesFunc);
	    size_t attVolDomainSizeFlat = std::accumulate(attVolDomainSize.begin(), attVolDomainSize.end(), size_t{1}, multiplesFunc);
        
	    std::vector<size_t> attPatchSliceSizes = compute_slice_sizes(attPatchSize);
        Index4d patchXSliceSizes = compute_slice_sizes_fast<DataNdim>(patchXSize);
        Index4d dataSliceSizes = compute_slice_sizes_fast<DataNdim>(dataSize);
        Index4d attVolDomainSliceSizes = compute_slice_sizes_fast<DataNdim>(attVolDomainSize);

        std::vector<float_t> attPatchRaw(attPatchSizeFlat);
        std::vector<float_t> attPatchScaled(patchXSizeFlat);

        std::vector<size_t> attIndexVec(DataNdim, 0);
        for (size_t attDomainIndexFlat = 0; attDomainIndexFlat < attVolDomainSizeFlat; attDomainIndexFlat++)
        {
            Index4d attIndexNd{};
            Index4d dataIndexNd{};
            unflattenIndex_fast(attDomainIndexFlat, attVolDomainSliceSizes, attIndexNd);

            std::copy(attIndexNd.begin(), attIndexNd.end(), attIndexVec.data());  // Convert to vector.

            // Compute the data index of the lower patch corner. Att volume can be smaller in the case of strided prediction.
            for (size_t dim = 0; dim < DataNdim; dim++)
                dataIndexNd[dim] = attIndexNd[dim] * predictionStride[dim];

            if (attIndexNd[1] == 0 && attIndexNd[2] == 0 && attIndexNd[3] == 0)
            {
                auto time = std::chrono::system_clock::now();
                std::time_t timeC = std::chrono::system_clock::to_time_t(time);
                std::string timeStr{std::ctime(&timeC)};
                timeStr.pop_back();

                printf("[%s] Processing frame %zu / %zu. \n", timeStr.c_str(), attIndexNd[0], attVolSize[0]);
                std::cout.flush();
            }

            // Read the raw attention patch.
            pAttentionRaw->ReadSlice(attIndexVec, AttNdim - DataNdim, attPatchRaw.data());
            // Upscale it to match the data patch size.
            upscale_attention_patch(attPatchRaw.data(), attPatchSize, attPatchSliceSizes,
                                    patchXSize, patchXSliceSizes, attPatchScaled);

            size_t firstIndexFlat = flattenIndex_fast(dataIndexNd, dataSliceSizes);
            size_t lastIndexFlat = flattenIndex_fast(dataIndexNd + patchXSizeNd, dataSliceSizes);

            pAttentionOut->_assureRangeInBuffer(firstIndexFlat, lastIndexFlat);

            for (size_t patchIndexFlat = 0; patchIndexFlat < patchXSizeFlat; patchIndexFlat++)
            {
                Index4d patchIndexNd{};
                unflattenIndex_fast(patchIndexFlat, patchXSliceSizes, patchIndexNd);
                size_t outputIndexFlat = flattenIndex_fast(dataIndexNd + patchIndexNd, dataSliceSizes);
                size_t relIndexFlat = outputIndexFlat - pAttentionOut->_bufferOffset;

                pAttentionOut->_buffer[relIndexFlat] = pAttentionOut->_buffer[relIndexFlat] + attPatchScaled[patchIndexFlat];
                pAttentionOut->_isBufferDirty = true;
            }
        }

        printf("Input buffer efficiency: %f\n", pAttentionRaw->ComputeBufferEfficiency());
        printf("Output buffer efficiency: %f\n", pAttentionOut->ComputeBufferEfficiency());
        std::cout.flush();
	}

    ///
    /// A dumb version of attention aggregation that works much faster.
    /// Used for debugging purposes.
    ///
    __DLLEXPORT
    void aggregate_attention_volume_dumb(void* pAttentionRawArray,
                                         size_t* pDataSize,
                                         size_t* pPatchSize,
                                         size_t predictionDelay,
                                         void* pAttentionOutArray)
    {
        const size_t dataNdim = 4;
        const size_t attNdim = 8;

        auto pAttentionRaw = reinterpret_cast<BufferedNdArray<float_t>*>(pAttentionRawArray);
        auto pAttentionOut = reinterpret_cast<BufferedNdArray<float_t>*>(pAttentionOutArray);

        std::vector<size_t> dataSize{ pDataSize, pDataSize + dataNdim };
        std::vector<size_t> patchSize{ pPatchSize, pPatchSize + dataNdim };
        std::vector<size_t> attVolSize = pAttentionRaw->GetShape();

        std::vector<size_t> attPatchSize{ attVolSize[4], attVolSize[5], attVolSize[6], attVolSize[7] };

        auto multiplesFunc = std::multiplies<>();
        size_t attPatchSizeFlat = std::accumulate(attPatchSize.begin(), attPatchSize.end(), size_t{1}, multiplesFunc);
        size_t patchSizeFlat = std::accumulate(patchSize.begin(), patchSize.end(), size_t{1}, multiplesFunc);
        size_t dataSizeFlat = std::accumulate(dataSize.begin(), dataSize.end(), size_t{1}, multiplesFunc);

        Index4d dataSliceSizes = compute_slice_sizes_fast<4>(dataSize);
        std::vector<size_t> attPatchSliceSizes = compute_slice_sizes(attPatchSize);
        std::vector<size_t> attVolSliceSizesVec = compute_slice_sizes(attVolSize);

        std::vector<float_t> attPatchRaw(attPatchSizeFlat);
        std::vector<float_t> attPatchScaled(patchSizeFlat);

        std::vector<size_t> domainLow{
            patchSize[0] - 2 + predictionDelay,
            patchSize[1] / 2,
            patchSize[2] / 2,
            patchSize[3] / 2
        };
        std::vector<size_t> domainHigh{
            dataSize[0],
            dataSize[1] - (patchSize[1] - patchSize[1] / 2) + 1,
            dataSize[2] - (patchSize[2] - patchSize[2] / 2) + 1,
            dataSize[3] - (patchSize[3] - patchSize[3] / 2) + 1
        };
        std::vector<size_t> dataIndexVec(dataNdim, 0);
        for (size_t dataIndexFlat = 0; dataIndexFlat < dataSizeFlat; dataIndexFlat++)
        {
            Index4d dataIndexNd{};
            unflattenIndex_fast(dataIndexFlat, dataSliceSizes, dataIndexNd);
            std::copy(dataIndexNd.X, dataIndexNd.X + dataNdim, dataIndexVec.data());  // Convert to vector.

            if (dataIndexNd.X[1] == 0 && dataIndexNd.X[2] == 0 && dataIndexNd.X[3] == 0)
                printf("Processing frame %zu. \n", dataIndexNd.X[0]);

            if (dataIndexNd.X[0] < domainLow[0] || dataIndexNd.X[0] >= domainHigh[0] ||
                dataIndexNd.X[1] < domainLow[1] || dataIndexNd.X[1] >= domainHigh[1] ||
                dataIndexNd.X[2] < domainLow[2] || dataIndexNd.X[2] >= domainHigh[2] ||
                dataIndexNd.X[3] < domainLow[3] || dataIndexNd.X[3] >= domainHigh[3])
            {
                continue;
            }

            // Read the raw attention patch.
            pAttentionRaw->ReadSlice(dataIndexVec, attNdim - dataNdim, attPatchRaw.data());
//            size_t attentionPatchIndexFlat = attPatchRaw.size() / 2;
            size_t attentionPatchIndexFlat = 0;

//            printf("%f \n", attPatchRaw[0]);
            float_t oldValue = pAttentionOut->Read(dataIndexFlat);
            pAttentionOut->Write(dataIndexFlat, oldValue + attPatchRaw[attentionPatchIndexFlat]);
        }

        printf("Input buffer efficiency: %f\n", pAttentionRaw->ComputeBufferEfficiency());
        printf("Output buffer efficiency: %f\n", pAttentionOut->ComputeBufferEfficiency());
    }


    ///
    ///
    ///
    __DLLEXPORT
    void aggregate_attention_volume_local_attention(void* pAttentionRawArray,
                                                    double_t* pOutAttentionAvg,
                                                    double_t* pOutAttentionVar)
    {
        constexpr size_t DataNdim = 4;
        constexpr size_t AttNdim = 8;

        auto pAttentionRaw = reinterpret_cast<BufferedNdArray<float_t>*>(pAttentionRawArray);

        std::vector<size_t> attVolSize = pAttentionRaw->GetShape();
        // Domain size includes only the spatiotemporal dimensions.
        std::vector<size_t> attVolDomainSize{ attVolSize.data(), attVolSize.data() + DataNdim };

        std::vector<size_t> attPatchSize{ attVolSize.begin() + DataNdim, attVolSize.end() };

        auto multiplesFunc = std::multiplies<>();
        size_t attPatchSizeFlat = std::accumulate(attPatchSize.begin(), attPatchSize.end(), size_t{1}, multiplesFunc);
        size_t attVolDomainSizeFlat = std::accumulate(attVolDomainSize.begin(), attVolDomainSize.end(), size_t{1}, multiplesFunc);
        Index4d attVolDomainSliceSizes = compute_slice_sizes_fast<DataNdim>(attVolDomainSize);

        // Zero-fill the output buffers to be safe.
        memset(pOutAttentionAvg, 0, attPatchSizeFlat);
        memset(pOutAttentionVar, 0, attPatchSizeFlat);

        std::vector<float_t> attPatchRaw(attPatchSizeFlat);
        std::vector<size_t> attIndexVec(DataNdim, 0);
        for (size_t attDomainIndexFlat = 0; attDomainIndexFlat < attVolDomainSizeFlat; attDomainIndexFlat++)
        {
            Index4d attIndexNd{};
            unflattenIndex_fast(attDomainIndexFlat, attVolDomainSliceSizes, attIndexNd);

            std::copy(attIndexNd.begin(), attIndexNd.end(), attIndexVec.data());  // Convert to vector.


            // Read the raw attention patch.
            pAttentionRaw->ReadSlice(attIndexVec, AttNdim - DataNdim, attPatchRaw.data());

            // For each voxel of the attention patch, add its value to the avg. patch buffer.
            for (size_t patchIndexFlat = 0; patchIndexFlat < attPatchSizeFlat; patchIndexFlat++)
            {
                double_t oldValue = *(pOutAttentionAvg + patchIndexFlat);
                *(pOutAttentionAvg + patchIndexFlat) = oldValue + attPatchRaw[patchIndexFlat];
            }
        }
        
	    // Now that we have a sum of all attention patches, we can compute the average by dividing.
        // For each voxel of the attention patch, add its value to the avg. patch buffer.
        for (size_t patchIndexFlat = 0; patchIndexFlat < attPatchSizeFlat; patchIndexFlat++)
        {
            double_t sum = *(pOutAttentionAvg + patchIndexFlat);
            *(pOutAttentionAvg + patchIndexFlat) = sum / static_cast<double_t>(attVolDomainSizeFlat);
        }

        // Repeat the same loop over the attention patches, now computing their variance.
        for (size_t attDomainIndexFlat = 0; attDomainIndexFlat < attVolDomainSizeFlat; attDomainIndexFlat++)
        {
            Index4d attIndexNd{};
            unflattenIndex_fast(attDomainIndexFlat, attVolDomainSliceSizes, attIndexNd);

            std::copy(attIndexNd.begin(), attIndexNd.end(), attIndexVec.data());  // Convert to vector.
            // Read the raw attention patch.
            pAttentionRaw->ReadSlice(attIndexVec, AttNdim - DataNdim, attPatchRaw.data());

            // For each voxel of the attention patch, add its value to the avg. patch buffer.
            for (size_t patchIndexFlat = 0; patchIndexFlat < attPatchSizeFlat; patchIndexFlat++)
            {
                double_t oldValue = *(pOutAttentionVar + patchIndexFlat);
                double_t mean = *(pOutAttentionAvg + patchIndexFlat);
                // Sum average square deviation from the mean.
                *(pOutAttentionVar + patchIndexFlat) = oldValue + std::pow(attPatchRaw[patchIndexFlat] - mean, 2);
            }
        }

        // Divide by the number of patches to get variance.
        for (size_t patchIndexFlat = 0; patchIndexFlat < attPatchSizeFlat; patchIndexFlat++)
        {
            double_t sum = *(pOutAttentionVar + patchIndexFlat);
            *(pOutAttentionVar + patchIndexFlat) = sum / static_cast<double_t>(attVolDomainSizeFlat);
        }
        
    }
}

