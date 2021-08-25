#pragma once
#include <vector>
#include <cassert>
#include <numeric>
#include <functional>
#include <algorithm>

size_t flattenIndex(std::vector<size_t> const& indexNd, std::vector<size_t> const& shape);
std::vector<size_t> unflattenIndex(size_t indexFlat, std::vector<size_t> const& shape);

std::vector<size_t> compute_slice_sizes(std::vector<size_t> const& shape);

size_t flattenIndex_fast(std::vector<size_t> const& indexNd, std::vector<size_t> const& sliceSizes);
void unflattenIndex_fast(size_t indexFlat, std::vector<size_t> const& sliceSizes, std::vector<size_t>& outIndexNd);

template<size_t TNdim>
struct IndexNd
{
    constexpr static size_t Ndim = TNdim;
    size_t X[TNdim];

    IndexNd()
    {
        std::fill(X, X + TNdim, 0);
    }

    template<typename InputIt>
    IndexNd(InputIt first, InputIt last)
    {
        assert(last - first == TNdim);
        std::copy(first, last, X);
    };

    // todo Use variadic template?
    IndexNd(std::initializer_list<size_t> const& list)
    {
        assert(list.size() == TNdim);
        std::copy(list.begin(), list.end(), X);
    }

    size_t& operator[] (size_t index)
    {
        assert(index >= 0);
        assert(index < TNdim);

        return X[index];
    }

    size_t const& operator[] (size_t index) const
    {
        assert(index >= 0);
        assert(index < TNdim);

        return X[index];
    }

    static size_t dot(IndexNd<TNdim> const& left, IndexNd<TNdim> const& right)
    {
        size_t result{0};

        for (size_t dim = 0; dim < TNdim; dim++)
            result += left.X[dim] * right.X[dim];

        return result;
    }

    size_t* begin() { return X; }
    size_t const* begin() const { return X; }
    size_t* end() { return X + TNdim; }
    size_t const* end() const { return X + TNdim; }
};

template<size_t TNdim>
IndexNd<TNdim> inline operator+ (IndexNd<TNdim> const& left, IndexNd<TNdim> const& right)
{
    IndexNd<TNdim> result{};
    for (size_t dim = 0; dim < TNdim; dim++)
        result.X[dim] = left.X[dim] + right.X[dim];

    return result;
}

template<size_t TNdim>
IndexNd<TNdim> inline operator* (IndexNd<TNdim> const& left, IndexNd<TNdim> const& right)
{
    IndexNd<TNdim> result{};
    for (size_t dim = 0; dim < TNdim; dim++)
        result.X[dim] = left.X[dim] * right.X[dim];

    return result;
}

template<size_t TNdim>
IndexNd<TNdim> vector_to_index(std::vector<size_t> const& shape)
{
    assert(shape.size() == TNdim);

    IndexNd<TNdim> index{};
    std::copy(shape.begin(), shape.end(), index.X);
    return index;
}

template<size_t TNdim>
std::vector<size_t> index_to_vector(IndexNd<TNdim> const& index)
{
    std::vector<size_t> shape(TNdim);
    std::copy(index.begin(), index.end(), shape.data());
    return shape;
}

template<size_t TNdim>
size_t flattenIndex_fast(IndexNd<TNdim> const& indexNd, IndexNd<TNdim> const& sliceSizes)
{
    size_t flatIndex = 0;
    for (size_t dim = 0; dim < TNdim; ++dim)
        flatIndex += indexNd.X[dim] * sliceSizes.X[dim];

    return flatIndex;
}

template<size_t TNdim>
void unflattenIndex_fast(size_t indexFlat, IndexNd<TNdim> const& sliceSizes, IndexNd<TNdim>& outIndexNd)
{
    for (size_t dim = 0; dim < TNdim; dim++)
    {
        outIndexNd[dim] = indexFlat / sliceSizes[dim];
        indexFlat = indexFlat % sliceSizes[dim];
    }
}

template<size_t TNdim>
IndexNd<TNdim> compute_slice_sizes_fast(IndexNd<TNdim> const& shape)
{
    IndexNd<TNdim> sliceSizes;
    sliceSizes[TNdim - 1] = 1;

    for (long dim = TNdim - 2; dim >= 0; --dim)
        sliceSizes[dim] = shape[dim + 1] * sliceSizes[dim + 1];

    return sliceSizes;
}

template<size_t TNdim>
IndexNd<TNdim> compute_slice_sizes_fast(std::vector<size_t> const& shape)
{
    IndexNd<TNdim> shapeFixed = vector_to_index<TNdim>(shape);
    return compute_slice_sizes_fast<TNdim>(shapeFixed);
}


typedef IndexNd<4> Index4d;


[[deprecated("Still uses patch size counting the extra y-frame.")]]
std::vector<size_t> compute_patch_number_old(const std::vector<size_t>& volumeSize,
                                             const std::vector<size_t>& patchSize,
                                             const std::vector<size_t>& patchStride,
                                             const std::vector<size_t>& patchInnerStride,
                                             size_t predictionDelay);


std::vector<size_t> compute_patch_number(const std::vector<size_t>& volumeSize,
                                         const std::vector<size_t>& patchSizeX, 
                                         const std::vector<size_t>& patchSizeY, 
                                         const std::vector<size_t>& patchStride,
                                         const std::vector<size_t>& patchInnerStride, 
                                         size_t predictionDelay);

/// Convolves a 3D array with a 1D averaging kernel along one of its axes.
/// Truncates the kernel near the borders.
template<typename TData, size_t TAxis>
void smooth_3d_array_average_1d_pass(TData const* pInputData, IndexNd<3> const& dataSize, size_t kernelRadius, TData* pOutputData)
{
    // Outer axes are the ones being iterated over untouched.
    // The inner axis is the one along which the kernel is applied.
    // Axis to outer axes: X -> ZY, Y -> ZX, Z -> YX 
    constexpr size_t axisOuter1 = TAxis != 0 ? 0 : 1;
    constexpr size_t axisOuter2 = TAxis != 2 ? 2 : 1;
    constexpr size_t axisInner  = TAxis;

    constexpr size_t ndim = 3;

    IndexNd<ndim> sliceSizes = compute_slice_sizes_fast<ndim>(dataSize);

    IndexNd<ndim> index{ 0, 0, 0 };
    IndexNd<ndim> tempIndex{ 0, 0, 0 }; // For storing modified index without re-allocation.
    for (index[axisOuter1] = 0; index[axisOuter1] < dataSize[axisOuter1]; ++index[axisOuter1])
    {
        for (index[axisOuter2] = 0; index[axisOuter2] < dataSize[axisOuter2]; ++index[axisOuter2])
        {
            double sum = 0.0;
            // Protect again the kernel being larger than the data.
            size_t preaggregatedCount = std::min(kernelRadius, dataSize[axisInner]);
            size_t count = preaggregatedCount;  // Fix the count to be (2r + 1), if you want to switch to zero-padded convolution.

            // Sum up half-a-kernel worth of elements before sliding.
            for (index[axisInner] = 0; index[axisInner] < preaggregatedCount; ++index[axisInner])
                sum += pInputData[flattenIndex_fast(index, sliceSizes)];

            // Slide along the axis, adding and dropping elements as needed.
            for (index[axisInner] = 0; index[axisInner] < dataSize[axisInner]; ++index[axisInner])
            {
                if (index[axisInner] >= kernelRadius + 1)
                {
                    tempIndex = index;
                    tempIndex[axisInner] -= kernelRadius + 1;
                    sum -= pInputData[flattenIndex_fast(tempIndex, sliceSizes)];
                    --count;
                }

                if ((index[axisInner] + kernelRadius) < dataSize[axisInner])
                {
                    tempIndex = index;
                    tempIndex[axisInner] += kernelRadius;
                    sum += pInputData[flattenIndex_fast(tempIndex, sliceSizes)];
                    ++count;
                }
                // Write the current value.
                pOutputData[flattenIndex_fast(index, sliceSizes)] = static_cast<TData>(sum / count);
            }
        }
    }
}

template<typename T>
void smooth_3d_array_average(T const* pInputData, IndexNd<3> const& dataSize, size_t kernelRadius, T* pOutputData)
{
    size_t dataSizeFlat = std::accumulate(dataSize.begin(), dataSize.end(), size_t{1}, std::multiplies<>());

    // Averaging cannot work in-place, use another buffer for intermediate calculations.
    std::vector<T> tempData(dataSizeFlat);

    smooth_3d_array_average_1d_pass<T, 2>(pInputData,  dataSize, kernelRadius, pOutputData);
    smooth_3d_array_average_1d_pass<T, 1>(pOutputData, dataSize, kernelRadius, tempData.data());
    smooth_3d_array_average_1d_pass<T, 0>(tempData.data(),   dataSize, kernelRadius, pOutputData);
}