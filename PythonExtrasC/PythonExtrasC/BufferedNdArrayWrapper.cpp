#include <cmath>
#include <cstdint>

#include "BufferedNdArray.hpp"
#include "macros.h"


template<typename TData>
BufferedNdArray<TData>* _cast_bna_pointer(void* pArrayRaw)
{
    if (pArrayRaw == nullptr)
        throw std::runtime_error("Null BufferedNdArray pointer provided.");

    return reinterpret_cast<BufferedNdArray<TData>*>(pArrayRaw);
}

template<typename TData>
void* bna_construct(wchar_t* pFilepath, uint8_t fileModeRaw, size_t const* pShape, size_t ndim, size_t maxBufferSizeFlat)
{
    std::wstring filepath{ pFilepath };
    typename BufferedNdArray<TData>::Tuple shape{ pShape, pShape + ndim };

    auto fileMode = static_cast<typename BufferedNdArray<TData>::FileMode>(fileModeRaw);
    auto* pArray = new BufferedNdArray<TData>(filepath, fileMode, shape, maxBufferSizeFlat);

    return pArray;
}

template<typename TData>
void bna_destruct(void* pArrayRaw)
{
    BufferedNdArray<TData>* pArray = _cast_bna_pointer<TData>(pArrayRaw);
    delete pArray;
}

template<typename TData>
TData bna_read(void* pArrayRaw, size_t const* pIndex)
{
    BufferedNdArray<TData>* pArray = _cast_bna_pointer<TData>(pArrayRaw);
    typename BufferedNdArray<TData>::Tuple indexNd(pIndex, pIndex + pArray->GetNdim());
    return pArray->Read(indexNd);
}

template<typename TData>
void bna_read_slice(void* pArrayRaw, size_t const* pSliceIndex, size_t sliceNdim, TData* pOutput)
{
    BufferedNdArray<TData>* pArray = _cast_bna_pointer<TData>(pArrayRaw);
    size_t indexNdim = pArray->GetNdim() - sliceNdim;
    typename BufferedNdArray<TData>::Tuple sliceIndexNd(pSliceIndex, pSliceIndex + indexNdim);
    pArray->ReadSlice(sliceIndexNd, sliceNdim, pOutput);
}

template<typename TData>
void bna_read_slab(void* pArrayRaw, size_t const* pIndicesLow, size_t const* pIndicesHigh, TData* pOutput)
{
    BufferedNdArray<TData>* pArray = _cast_bna_pointer<TData>(pArrayRaw);
    typename BufferedNdArray<TData>::Tuple indicesLow(pIndicesLow, pIndicesLow + pArray->GetNdim());
    typename BufferedNdArray<TData>::Tuple indicesHigh(pIndicesHigh, pIndicesHigh + pArray->GetNdim());
    pArray->ReadSlab(indicesLow, indicesHigh, pOutput);
}

template<typename TData>
void bna_write(void* pArrayRaw, size_t const* pIndex, TData value)
{
    BufferedNdArray<TData>* pArray = _cast_bna_pointer<TData>(pArrayRaw);
    typename BufferedNdArray<TData>::Tuple indexNd(pIndex, pIndex + pArray->GetNdim());
    pArray->Write(indexNd, value);
}

template<typename TData>
void bna_write_slice(void* pArrayRaw, size_t const* pSliceIndex, size_t sliceNdim, TData const* pInput)
{
    BufferedNdArray<TData>* pArray = _cast_bna_pointer<TData>(pArrayRaw);
    size_t indexNdim = pArray->GetNdim() - sliceNdim;
    typename BufferedNdArray<TData>::Tuple sliceIndexNd(pSliceIndex, pSliceIndex + indexNdim);
    pArray->WriteSlice(sliceIndexNd, sliceNdim, pInput);
}

template<typename TData>
void bna_write_full(void* pArrayRaw, TData const* pInput)
{
    BufferedNdArray<TData>* pArray = _cast_bna_pointer<TData>(pArrayRaw);
    pArray->WriteFull(pInput);
}


template<typename TData>
void bna_write_patch(void* pArrayRaw, size_t const* pPatchLowIndex, size_t const* pPatchSize, TData const* pInput)
{
    BufferedNdArray<TData>* pArray = _cast_bna_pointer<TData>(pArrayRaw);
    typename BufferedNdArray<TData>::Tuple patchLowIndex(pPatchLowIndex, pPatchLowIndex + pArray->GetNdim());
    typename BufferedNdArray<TData>::Tuple patchSize(pPatchSize, pPatchSize + pArray->GetNdim());
    pArray->WritePatch(patchLowIndex, patchSize, pInput);
}


template<typename TData>
void bna_fill_box(void* pArrayRaw, TData value, size_t const* pCornerLow, size_t const* pCornerHigh)
{
    BufferedNdArray<TData>* pArray = _cast_bna_pointer<TData>(pArrayRaw);
    size_t ndim = pArray->GetNdim();
    pArray->FillBox(value,
                    typename BufferedNdArray<TData>::Tuple(pCornerLow, pCornerLow + ndim),
                    typename BufferedNdArray<TData>::Tuple(pCornerHigh, pCornerHigh + ndim));
}

template<typename TData>
void bna_set_direct_mode(void* pArrayRaw, bool isDirectMode)
{
    BufferedNdArray<TData>* pArray = _cast_bna_pointer<TData>(pArrayRaw);
    pArray->SetDirectMode(isDirectMode);
}

template<typename TData>
void bna_flush(void* pArrayRaw, bool flushOsBuffer)
{
    BufferedNdArray<TData>* pArray = _cast_bna_pointer<TData>(pArrayRaw);
    pArray->FlushBuffer(flushOsBuffer);
}

template<typename TData>
float_t bna_compute_buffer_efficiency(void* pArrayRaw)
{
    BufferedNdArray<TData>* pArray = _cast_bna_pointer<TData>(pArrayRaw);
    return pArray->ComputeBufferEfficiency();
}

template<typename TData>
void bna_reset_counters(void* pArrayRaw)
{
    BufferedNdArray<TData>* pArray = _cast_bna_pointer<TData>(pArrayRaw);
    pArray->ResetCounters();
}

extern "C" {

    __DLLEXPORT
    void* bna_construct_uint8(wchar_t* pFilepath, uint8_t fileMode, size_t const* pShape, size_t ndim, size_t maxBufferSizeFlat)
    {
        return bna_construct<uint8_t>(pFilepath, fileMode, pShape, ndim, maxBufferSizeFlat);
    }
    __DLLEXPORT
    void* bna_construct_float32(wchar_t* pFilepath, uint8_t fileMode, size_t const* pShape, size_t ndim, size_t maxBufferSizeFlat)
    {
        return bna_construct<float_t>(pFilepath, fileMode, pShape, ndim, maxBufferSizeFlat);
    }

    __DLLEXPORT
    void bna_destruct_uint8(void* pArrayRaw)
    {
        bna_destruct<uint8_t>(pArrayRaw);
    }
    __DLLEXPORT
    void bna_destruct_float32(void* pArrayRaw)
    {
        bna_destruct<float_t>(pArrayRaw);
    }

    __DLLEXPORT
    uint8_t bna_read_uint8(void* pArrayRaw, size_t const* pIndex)
    {
        return bna_read<uint8_t>(pArrayRaw, pIndex);
    }
    __DLLEXPORT
    float_t bna_read_float32(void* pArrayRaw, size_t const* pIndex)
    {
        return bna_read<float_t>(pArrayRaw, pIndex);
    }

    __DLLEXPORT
    void bna_read_slice_uint8(void* pArrayRaw, size_t const* pSliceIndex, size_t sliceNdim, uint8_t* pOutput)
    {
        bna_read_slice<uint8_t>(pArrayRaw, pSliceIndex, sliceNdim, pOutput);
    }
    __DLLEXPORT
    void bna_read_slice_float32(void* pArrayRaw, size_t const* pSliceIndex, size_t sliceNdim,  float_t* pOutput)
    {
        bna_read_slice<float_t>(pArrayRaw, pSliceIndex, sliceNdim, pOutput);
    }

    __DLLEXPORT
    void bna_read_slab_uint8(void* pArrayRaw, size_t const* pIndicesLow, size_t const* pIndicesHigh, uint8_t* pOutput)
    {
        bna_read_slab<uint8_t>(pArrayRaw, pIndicesLow, pIndicesHigh, pOutput);
    }
    __DLLEXPORT
    void bna_read_slab_float32(void* pArrayRaw, size_t const* pIndicesLow, size_t const* pIndicesHigh, float_t* pOutput)
    {
        bna_read_slab<float_t>(pArrayRaw, pIndicesLow, pIndicesHigh, pOutput);
    }

    __DLLEXPORT
    void bna_write_uint8(void* pArrayRaw, size_t const* pIndex, uint8_t value)
    {
        bna_write<uint8_t>(pArrayRaw, pIndex, value);
    }
    __DLLEXPORT
    void bna_write_float32(void* pArrayRaw, size_t const* pIndex, float_t value)
    {
        bna_write<float_t>(pArrayRaw, pIndex, value);
    }

    __DLLEXPORT
    void bna_write_slice_uint8(void* pArrayRaw, size_t const* pSliceIndex, size_t sliceNdim, uint8_t const* pInput)
    {
        bna_write_slice<uint8_t>(pArrayRaw, pSliceIndex, sliceNdim, pInput);
    }
    __DLLEXPORT
    void bna_write_slice_float32(void* pArrayRaw, size_t const* pSliceIndex, size_t sliceNdim, float_t const* pInput)
    {
        bna_write_slice<float_t>(pArrayRaw, pSliceIndex, sliceNdim, pInput);
    }

    __DLLEXPORT
    void bna_write_full_uint8(void* pArrayRaw, uint8_t const* pInput)
    {
        bna_write_full<uint8_t>(pArrayRaw, pInput);
    }
    __DLLEXPORT
    void bna_write_full_float32(void* pArrayRaw, float_t const* pInput)
    {
        bna_write_full<float_t>(pArrayRaw, pInput);
    }
    
    __DLLEXPORT
    void bna_write_patch_uint8(void* pArrayRaw, size_t const* pPatchLowIndex, size_t const* pPatchSize, uint8_t const* pInput)
    {
        bna_write_patch(pArrayRaw, pPatchLowIndex, pPatchSize, pInput);
    }
    __DLLEXPORT
    void bna_write_patch_float32(void* pArrayRaw, size_t const* pPatchLowIndex, size_t const* pPatchSize, float_t const* pInput)
    {
        bna_write_patch(pArrayRaw, pPatchLowIndex, pPatchSize, pInput);
    }

    __DLLEXPORT
    void bna_fill_box_uint8(void* pArrayRaw, uint8_t value, size_t* pCornerLow, size_t* pCornerHigh)
    {
        bna_fill_box<uint8_t>(pArrayRaw, value, pCornerLow, pCornerHigh);
    }
    __DLLEXPORT
    void bna_fill_box_float32(void* pArrayRaw, float_t value, size_t* pCornerLow, size_t* pCornerHigh)
    {
        bna_fill_box<float_t>(pArrayRaw, value, pCornerLow, pCornerHigh);
    }

    __DLLEXPORT
    void bna_set_direct_mode_uint8(void* pArrayRaw, bool isDirectMode)
    {
        bna_set_direct_mode<uint8_t>(pArrayRaw, isDirectMode);
    }
    __DLLEXPORT
    void bna_set_direct_mode_float32(void* pArrayRaw, bool isDirectMode)
    {
        bna_set_direct_mode<float_t>(pArrayRaw, isDirectMode);
    }

    __DLLEXPORT
    void bna_flush_uint8(void* pArrayRaw, bool flushOsBuffer)
    {
        bna_flush<uint8_t>(pArrayRaw, flushOsBuffer);
    }
    __DLLEXPORT
    void bna_flush_float32(void* pArrayRaw, bool flushOsBuffer)
    {
        bna_flush<float_t>(pArrayRaw, flushOsBuffer);
    }

    __DLLEXPORT
    float_t bna_compute_buffer_efficiency_uint8(void* pArrayRaw)
    {
        return bna_compute_buffer_efficiency<uint8_t>(pArrayRaw);
    }
    __DLLEXPORT
    float_t bna_compute_buffer_efficiency_float32(void* pArrayRaw)
    {
        return bna_compute_buffer_efficiency<float_t>(pArrayRaw);
    }

    __DLLEXPORT
    void bna_reset_counters_uint8(void* pArrayRaw)
    {
        bna_reset_counters<uint8_t>(pArrayRaw);
    }
    __DLLEXPORT
    void bna_reset_counters_float32(void* pArrayRaw)
    {
        bna_reset_counters<uint8_t>(pArrayRaw);
    }
}
