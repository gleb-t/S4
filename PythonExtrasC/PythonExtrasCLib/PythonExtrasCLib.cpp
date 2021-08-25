#include "PythonExtrasCLib.h"

#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <functional>
#include <random>
#include <thread>
#include <atomic>
#include <stdexcept>

typedef struct { size_t x[4]; } index4d_t;  // Why is this a typedef and not a normal struct? Change.


size_t flattenIndex(const std::vector<size_t>& indexNd, const std::vector<size_t>& shape)
{
    size_t ndim = shape.size();
    std::vector<size_t> sliceSizes(ndim);
    sliceSizes[ndim - 1] = 1;

    for (long dim = static_cast<long>(ndim - 2); dim >= 0; dim--) // Use signed int, because dim becomes '-1' at the end.
        sliceSizes[dim] = shape[dim + 1] * sliceSizes[dim + 1];

    size_t flatIndex = 0;
    for (size_t dim = 0; dim < ndim; dim++)
        flatIndex += indexNd[dim] * sliceSizes[dim];

    return flatIndex;
}

///
/// Convert a flat index into an N-d index (a tuple).
///
std::vector<size_t> unflattenIndex(size_t indexFlat, const std::vector<size_t>& shape)
{
    size_t ndim = shape.size();
    std::vector<size_t> indexNd(ndim, 0);
    auto multiplies = std::multiplies<>();
    for (size_t dim = 0; dim < ndim; dim++)
    {
        size_t sliceSize = std::accumulate(shape.begin() + dim + 1, shape.end(), size_t{1}, multiplies);
        size_t axisIndex = indexFlat / sliceSize;
        indexFlat -= axisIndex * sliceSize;
        indexNd[dim] = axisIndex;
    }

    return indexNd;
}

std::vector<size_t> compute_slice_sizes(const std::vector<size_t>& shape)
{
    size_t ndim = shape.size();
    std::vector<size_t> sliceSizes(ndim);
    sliceSizes[ndim - 1] = 1;

    for (long dim = static_cast<long>(ndim - 2); dim >= 0; dim--)
        sliceSizes[dim] = shape[dim + 1] * sliceSizes[dim + 1];

    return sliceSizes;
}

///
/// A fast version of the original 'flattenIndex'.
/// Uses a precomputed 'sliceSizes' vector obtained from 'compute_slice_sizes'.
///
size_t flattenIndex_fast(std::vector<size_t> const& indexNd, std::vector<size_t> const& sliceSizes)
{
    size_t flatIndex = 0;
    for (size_t dim = 0; dim < indexNd.size(); dim++)
        flatIndex += indexNd[dim] * sliceSizes[dim];

    return flatIndex;
}

///
/// A fast version of the original 'unflattenIndex'.
/// Uses a precomputed 'sliceSizes' vector obtained from 'compute_slice_sizes'.
///
void unflattenIndex_fast(size_t indexFlat, std::vector<size_t> const& sliceSizes, std::vector<size_t>& outIndexNd)
{
    for (size_t dim = 0; dim < sliceSizes.size(); dim++)
    {
        outIndexNd[dim] = indexFlat / sliceSizes[dim];
        indexFlat = indexFlat % sliceSizes[dim];
    }
}


///
/// Compute the number of patches along each dimension.
///
std::vector<size_t> compute_patch_number_old(const std::vector<size_t>& volumeSize,
                                             const std::vector<size_t>& patchSize, 
                                             const std::vector<size_t>& patchStride,
                                             const std::vector<size_t>& patchInnerStride, 
                                             size_t predictionDelay)
{
    std::vector<size_t> patchNumber(volumeSize.size());
    for (size_t i = 0; i < volumeSize.size(); i++)
    {
        size_t stride = patchStride[i];
        // How many voxels a patch covers.
        // Last point in time (Y-value) is 'predictionDelay' frames away from the previous frame.
        // I.e. if 'predictionDelay' is 1, it immediately follows it.
        size_t patchSupport = i > 0 ? (patchSize[i] - 1) * patchInnerStride[i] + 1
                                    : (patchSize[i] - 2) * patchInnerStride[i] + 1 + predictionDelay;
        size_t totalPatchNumber = volumeSize[i] - patchSupport + 1;
        patchNumber[i] = (totalPatchNumber + stride - 1) / stride;  // Round up.
    }

    return patchNumber;
}


///
/// Compute the number of patches along each dimension.
/// This version uses output patches instead of a single-voxel Y value.
///
std::vector<size_t> compute_patch_number(const std::vector<size_t>& volumeSize,
                                         const std::vector<size_t>& patchSizeX, 
                                         const std::vector<size_t>& patchSizeY, 
                                         const std::vector<size_t>& patchStride,
                                         const std::vector<size_t>& patchInnerStride, 
                                         size_t predictionDelay)
{
    std::vector<size_t> patchNumber(volumeSize.size());
    for (size_t i = 0; i < volumeSize.size(); i++)
    {
        if (patchSizeX[i] < patchSizeY[i])
        {
            printf("Output patch is expected to be smaller or equal to the input patch! \n");
            throw std::runtime_error("Output patch is expected to be smaller than the input patch! \n");
        }

        size_t stride = patchStride[i];
        // How many voxels a patch covers.
        // The Y-patch is 'predictionDelay' frames away from the last frame in the X-patch.
        // I.e. if 'predictionDelay' is 1, it immediately follows it.
        size_t patchSupport = i > 0 ? (patchSizeX[i] - 1) * patchInnerStride[i] + 1
                                    : (patchSizeX[i] - 1) * patchInnerStride[i] + patchSizeY[i] + predictionDelay;
        size_t totalPatchNumber = volumeSize[i] - patchSupport + 1;
        patchNumber[i] = (totalPatchNumber + stride - 1) / stride;  // Round up.
    }

    return patchNumber;
}

