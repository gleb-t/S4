#include "MtPatchExtractor.hpp"
#include "macros.h"


template<typename TData>
void* mtpe_construct(void* pVolumeData,
                     size_t const* pVolumeSize,
                     size_t featureNumber,
                     size_t const* pPatchXSize,
                     size_t const* pPatchYSize,
                     size_t const* pPatchStride,
                     size_t const* pPatchInnerStride,
                     size_t predictionDelay,
                     bool detectEmptyPatches,
                     TData emptyValue,
                     size_t emptyCheckFeature,

                     float_t undersamplingProbAny,
                     float_t undersamplingProbEmpty,
                     float_t undersamplingProbNonempty,

                     size_t inputBufferSize,
                     size_t threadNumber)
{

    auto* pMtpe = new MtPatchExtractor<TData>(
        reinterpret_cast<BufferedNdArray<TData>*>(pVolumeData),
        std::vector<size_t>(pVolumeSize, pVolumeSize + 4),
        featureNumber,
        std::vector<size_t>(pPatchXSize, pPatchXSize + 4),
        std::vector<size_t>(pPatchYSize, pPatchYSize + 4),
        std::vector<size_t>(pPatchStride, pPatchStride + 4),
        std::vector<size_t>(pPatchInnerStride, pPatchInnerStride + 4),
        predictionDelay,
        detectEmptyPatches,
        emptyValue,
        emptyCheckFeature,

        undersamplingProbAny,
        undersamplingProbEmpty,
        undersamplingProbNonempty,

        inputBufferSize,
        threadNumber);

    return pMtpe;
}


template<typename TData>
void mtpe_destruct(void* pMtpe)
{
    delete reinterpret_cast<MtPatchExtractor<TData>*>(pMtpe);
}

template<typename TData>
void mtpe_extract_batch(void* pMtpeRaw,
                        size_t batchStartIndex,
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
    auto* pMpte = reinterpret_cast<MtPatchExtractor<TData>*>(pMtpeRaw);
    pMpte->ExtractBatch(
        batchStartIndex,
        batchSize,

        pOutX,
        pOutY,
        pOutIndices,

        pOutPatchesChecked,
        pOutPatchesEmpty,
        pOutPatchesExtracted,
        pOutNextBatchIndex,
        pOutInputEndReached
    );
}

extern "C" {

    __DLLEXPORT
    void* mtpe_construct_uint8(void* pVolumeData,
                               size_t const* pVolumeSize,
                               size_t featureNumber,
                               size_t const* pPatchXSize,
                               size_t const* pPatchYSize,
                               size_t const* pPatchStride,
                               size_t const* pPatchInnerStride,
                               size_t predictionDelay,
                               bool detectEmptyPatches,
                               uint8_t emptyValue,
                               size_t emptyCheckFeature,

                               float_t undersamplingProbAny,
                               float_t undersamplingProbEmpty,
                               float_t undersamplingProbNonempty,

                               size_t inputBufferSize,
                               size_t threadNumber)
    {
        return mtpe_construct<uint8_t>(
            pVolumeData,
            pVolumeSize,
            featureNumber,
            pPatchXSize,
            pPatchYSize,
            pPatchStride,
            pPatchInnerStride,
            predictionDelay,
            detectEmptyPatches,
            emptyValue,
            emptyCheckFeature,

            undersamplingProbAny,
            undersamplingProbEmpty,
            undersamplingProbNonempty,

            inputBufferSize,
            threadNumber
        );
    }
    __DLLEXPORT
    void* mtpe_construct_float32(void* pVolumeData,
                                 size_t const* pVolumeSize,
                                 size_t featureNumber,
                                 size_t const* pPatchXSize,
                                 size_t const* pPatchYSize,
                                 size_t const* pPatchStride,
                                 size_t const* pPatchInnerStride,
                                 size_t predictionDelay,
                                 bool detectEmptyPatches,
                                 float_t emptyValue,
                                 size_t emptyCheckFeature,

                                 float_t undersamplingProbAny,
                                 float_t undersamplingProbEmpty,
                                 float_t undersamplingProbNonempty,

                                 size_t inputBufferSize,
                                 size_t threadNumber)
    {
        return mtpe_construct<float_t>(
            pVolumeData,
            pVolumeSize,
            featureNumber,
            pPatchXSize,
            pPatchYSize,
            pPatchStride,
            pPatchInnerStride,
            predictionDelay,
            detectEmptyPatches,
            emptyValue,
            emptyCheckFeature,

            undersamplingProbAny,
            undersamplingProbEmpty,
            undersamplingProbNonempty,

            inputBufferSize,
            threadNumber
            );
    }

    __DLLEXPORT
    void mtpe_extract_batch_uint8(void* pMtpeRaw,
                                  size_t batchStartIndex,
                                  size_t batchSize,

                                  uint8_t* pOutX,
                                  uint8_t* pOutY,
                                  size_t* pOutIndices,

                                  size_t* pOutPatchesChecked,
                                  size_t* pOutPatchesEmpty,
                                  size_t* pOutPatchesExtracted,
                                  size_t* pOutNextBatchIndex,
                                  bool* pOutInputEndReached)
    {
        mtpe_extract_batch<uint8_t>(
            pMtpeRaw,
            batchStartIndex,
            batchSize,

            pOutX,
            pOutY,
            pOutIndices,

            pOutPatchesChecked,
            pOutPatchesEmpty,
            pOutPatchesExtracted,
            pOutNextBatchIndex,
            pOutInputEndReached
        );
    }

    __DLLEXPORT
    void mtpe_extract_batch_float32(void* pMtpeRaw,
                                    size_t batchStartIndex,
                                    size_t batchSize,

                                    float_t* pOutX,
                                    float_t* pOutY,
                                    size_t* pOutIndices,

                                    size_t* pOutPatchesChecked,
                                    size_t* pOutPatchesEmpty,
                                    size_t* pOutPatchesExtracted,
                                    size_t* pOutNextBatchIndex,
                                    bool* pOutInputEndReached)
    {
        mtpe_extract_batch<float_t>(
            pMtpeRaw,
            batchStartIndex,
            batchSize,

            pOutX,
            pOutY,
            pOutIndices,

            pOutPatchesChecked,
            pOutPatchesEmpty,
            pOutPatchesExtracted,
            pOutNextBatchIndex,
            pOutInputEndReached
            );
    }

    __DLLEXPORT
    void mtpe_destruct_uint8(void* pMtpe)
    {
        mtpe_destruct<uint8_t>(pMtpe);
    }
    __DLLEXPORT
    void mtpe_destruct_float32(void* pMtpe)
    {
        mtpe_destruct<float_t>(pMtpe);
    }

}
