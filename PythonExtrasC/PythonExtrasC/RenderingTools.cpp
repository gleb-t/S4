#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <cstdint>
#include <cstddef>
#include <functional>
#include <cmath>
#include <vector>

#include "macros.h"
#include <numeric>


extern "C" {

    
    /**
    * \brief 
    * \param pData   A 4D array of uint8.
    * \param pDataSize 
    * \param pTransferFunc 
    * \param pResult 
    */
    __DLLEXPORT
    void apply_tf_to_volume_uint8(uint8_t const* pData,
                                  size_t const* pDataSize,
                                  uint8_t const* pTransferFunc,
        uint8_t* pResult)
    {
        constexpr size_t ndim = 4;
        constexpr size_t channels = 4;

        std::vector<size_t> dataSize{pDataSize, pDataSize + ndim};
        uint64_t dataSizeFlat = std::accumulate(dataSize.begin(), dataSize.end(),
                                                1, std::multiplies<size_t>());

        for (size_t indexFlat = 0; indexFlat < dataSizeFlat; indexFlat++)
        {
            uint8_t dataVal = pData[indexFlat];
            std::copy(pTransferFunc + dataVal * channels,
                      pTransferFunc + dataVal * channels + channels,
                      pResult + indexFlat * channels);
        }
    }
}
