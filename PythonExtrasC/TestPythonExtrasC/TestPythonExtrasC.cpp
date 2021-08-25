#include <numeric>
#include <fstream>


#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main()
#include "catch.hpp"

#include <fileapi.h>

#include "BufferedNdArray.hpp"
#include "PythonExtrasCLib.h"


TEST_CASE("Data persistence and integrity tests.", "[BufferedNdArray]") {

    // First, measure the path length, then construct a buffer and fetch the path.
    std::vector<wchar_t> tmpDirPathBuffer(GetTempPathW(0, nullptr));
    GetTempPathW(static_cast<DWORD>(tmpDirPathBuffer.size()), tmpDirPathBuffer.data());

    std::wstring dataPath{ tmpDirPathBuffer.begin(), tmpDirPathBuffer.end() - 1 }; // Last char is \0.
    dataPath += L"test-python-extras.raw";

    BufferedNdArray<uint8_t>::Tuple dataShape{ 7, 19, 20, 21 };
    size_t maxBufferSizeFlat = 2 * 19 * 20 * 21 + 6789;
    auto pNdarray = std::make_unique<BufferedNdArray<uint8_t>>(dataPath, BufferedNdArray<uint8_t>::FileMode::Rewrite,
                                                               dataShape, maxBufferSizeFlat);

    size_t dataSizeFlat = std::accumulate(dataShape.begin(), dataShape.end(), size_t{1}, 
                                          std::multiplies<>());
    for (size_t indexFlat = 0; indexFlat < dataSizeFlat; indexFlat++)
    {
        std::vector<size_t> indexNd = unflattenIndex(indexFlat, dataShape);
        
        // Compute a basic hash.
        uint8_t value = 1;
        for (size_t coord : indexNd)
            value = static_cast<uint8_t>((31 * value + coord) % 256);
//        value = indexFlat % 256;

        pNdarray->Write(indexNd, value);
    }

    pNdarray->FlushBuffer(true);

    SECTION("Data can be read back directly from the file.")
    {
        // Close the dataset.
        pNdarray.reset();

        std::fstream file{dataPath, std::ios::in | std::ios::binary};
        std::vector<uint8_t> buffer(dataSizeFlat);

        file.read(reinterpret_cast<char*>(buffer.data()), dataSizeFlat);
        file.close();

        bool allValuesEqual = true;
        for (size_t indexFlat = 0; indexFlat < dataSizeFlat; indexFlat++)
        {
            std::vector<size_t> indexNd = unflattenIndex(indexFlat, dataShape);

            // Compute a basic hash.
            uint8_t expectedValue = 1;
            for (size_t coord : indexNd)
                expectedValue = static_cast<uint8_t>((31 * expectedValue + coord) % 256);
//            expectedValue = indexFlat % 256;

            if (expectedValue != buffer[indexFlat])
            {
                printf("Expected %u, got %u instead at index %zu. \n", expectedValue, buffer[indexFlat], indexFlat);
                allValuesEqual = false;
                break;
            }
        }

        REQUIRE(allValuesEqual);
    }

    SECTION("Data can be read back using the same instance.")
    {
        bool allValuesEqual = true;
        for (size_t indexFlat = 0; indexFlat < dataSizeFlat; indexFlat++)
        {
            std::vector<size_t> indexNd = unflattenIndex(indexFlat, dataShape);

            // Compute a basic hash.
            uint8_t expectedValue = 1;
            for (size_t coord : indexNd)
                expectedValue = static_cast<uint8_t>((31 * expectedValue + coord) % 256);
            //            uint8_t expectedValue = indexFlat % 256;

            uint8_t actualValue = pNdarray->Read(indexNd);
            if (expectedValue != actualValue)
            {
                printf("Expected %u, got %u instead at index %zu. \n", expectedValue, actualValue, indexFlat);
                allValuesEqual = false;
                break;
            }
        }

        REQUIRE(allValuesEqual);
    }

    SECTION("Data can be read back using a new instance.")
    {
        pNdarray.reset();
        pNdarray = std::make_unique<BufferedNdArray<uint8_t>>(dataPath, BufferedNdArray<uint8_t>::FileMode::Update,
                                                              dataShape, maxBufferSizeFlat);
        bool allValuesEqual = true;
        for (size_t indexFlat = 0; indexFlat < dataSizeFlat; indexFlat++)
        {
            std::vector<size_t> indexNd = unflattenIndex(indexFlat, dataShape);

            // Compute a basic hash.
            uint8_t expectedValue = 1;
            for (size_t coord : indexNd)
                expectedValue = static_cast<uint8_t>((31 * expectedValue + coord) % 256);
            //            uint8_t expectedValue = indexFlat % 256;

            uint8_t actualValue = pNdarray->Read(indexNd);
            if (expectedValue != actualValue)
            {
                printf("Expected %u, got %u instead at index %zu. \n", expectedValue, actualValue, indexFlat);
                allValuesEqual = false;
                break;
            }
        }

        REQUIRE(allValuesEqual);
    }

    SECTION("Recreating the instance with Rewrite mode should recreate the file.")
    {
        pNdarray.reset();
        pNdarray = std::make_unique<BufferedNdArray<uint8_t>>(dataPath, BufferedNdArray<uint8_t>::FileMode::Rewrite,
                                                              dataShape, maxBufferSizeFlat);
        bool allValuesEqual = true;
        for (size_t indexFlat = 0; indexFlat < dataSizeFlat; indexFlat++)
        {
            std::vector<size_t> indexNd = unflattenIndex(indexFlat, dataShape);

            uint8_t expectedValue = 0;
            uint8_t actualValue = pNdarray->Read(indexNd);
            if (expectedValue != actualValue)
            {
                printf("Expected %u, got %u instead at index %zu. \n", expectedValue, actualValue, indexFlat);
                allValuesEqual = false;
                break;
            }
        }

        REQUIRE(allValuesEqual);
    }

    SECTION("FillBox Test.")
    {
        BufferedNdArray<uint8_t>::Tuple cornerLow{ 2, 2, 2, 2 };
        BufferedNdArray<uint8_t>::Tuple cornerHigh{ 4, 5, 6, 7 };
        const uint8_t fillValue = 137;
        pNdarray->FillBox(fillValue, cornerLow, cornerHigh);

        CHECK(pNdarray->Read(cornerLow) == fillValue);
        CHECK(pNdarray->Read(cornerHigh) != fillValue);
        CHECK(pNdarray->Read(BufferedNdArray<uint8_t>::Tuple{2, 2, 2, 3}) == fillValue);
        CHECK(pNdarray->Read(BufferedNdArray<uint8_t>::Tuple{3, 2, 2, 2}) == fillValue);
        CHECK(pNdarray->Read(BufferedNdArray<uint8_t>::Tuple{2, 2, 2, 1}) != fillValue);
        CHECK(pNdarray->Read(BufferedNdArray<uint8_t>::Tuple{2, 2, 2, 8}) != fillValue);
        CHECK(pNdarray->Read(BufferedNdArray<uint8_t>::Tuple{1, 2, 2, 2}) != fillValue);
        CHECK(pNdarray->Read(BufferedNdArray<uint8_t>::Tuple{5, 2, 2, 2}) != fillValue);

        CHECK(pNdarray->Read(BufferedNdArray<uint8_t>::Tuple{3, 4, 5, 6}) == fillValue);
        CHECK(pNdarray->Read(BufferedNdArray<uint8_t>::Tuple{2, 4, 5, 6}) == fillValue);
        CHECK(pNdarray->Read(BufferedNdArray<uint8_t>::Tuple{3, 4, 5, 5}) == fillValue);
        CHECK(pNdarray->Read(BufferedNdArray<uint8_t>::Tuple{5, 5, 6, 7}) != fillValue);
        CHECK(pNdarray->Read(BufferedNdArray<uint8_t>::Tuple{1, 5, 6, 7}) != fillValue);
        CHECK(pNdarray->Read(BufferedNdArray<uint8_t>::Tuple{4, 5, 6, 1}) != fillValue);
        CHECK(pNdarray->Read(BufferedNdArray<uint8_t>::Tuple{4, 5, 6, 8}) != fillValue);
    }

    SECTION("ReadSlice Test. First axis only.")
    {
        std::vector<size_t> sliceSizes = compute_slice_sizes(dataShape);

        std::vector<uint8_t> sliceBuffer(sliceSizes[0], 0);
        BufferedNdArray<uint8_t>::Tuple sliceIndexNd(1, 0);  // Vector with a single value.
        for (size_t sliceIndex = 0; sliceIndex < dataShape[0]; sliceIndex++)
        {
            sliceIndexNd[0] = sliceIndex;
            pNdarray->ReadSlice(sliceIndexNd, size_t{3}, sliceBuffer.data());

            bool allValuesEqual = true;
            for (size_t relIndexFlat = 0; relIndexFlat < sliceSizes[0]; relIndexFlat++)
            {
                size_t indexFlat = relIndexFlat + sliceIndex * sliceSizes[0];
                std::vector<size_t> indexNd = unflattenIndex(indexFlat, dataShape);
                // Compute a basic hash.
                uint8_t expectedValue = 1;
                for (size_t coord : indexNd)
                    expectedValue = static_cast<uint8_t>((31 * expectedValue + coord) % 256);

                uint8_t actualValue = sliceBuffer[relIndexFlat];
                if (expectedValue != actualValue)
                {
                    printf("Expected %u, got %u instead at index %zu. \n", expectedValue, actualValue, indexFlat);
                    allValuesEqual = false;
                    break;
                }
            }

            REQUIRE(allValuesEqual);
        }
    }
}

TEST_CASE("Large dataset size test.", "[BufferedNdArray]") {

    // First, measure the path length, then construct a buffer and fetch the path.
    std::vector<wchar_t> tmpDirPathBuffer(GetTempPathW(0, nullptr));
    GetTempPathW(static_cast<DWORD>(tmpDirPathBuffer.size()), tmpDirPathBuffer.data());

    std::wstring dataPath{ tmpDirPathBuffer.begin(), tmpDirPathBuffer.end() - 1 }; // Last char is \0.
    dataPath += L"test-python-extras-large-dataset.raw";

    BufferedNdArray<uint8_t>::Tuple dataShape{ 400, 256, 256, 256 };
    size_t maxBufferSizeFlat = 2 * 256 * 256 * 256 + 6789;
    auto pNdarray = std::make_unique<BufferedNdArray<uint8_t>>(dataPath, BufferedNdArray<uint8_t>::FileMode::Rewrite,
                                                               dataShape, maxBufferSizeFlat);

    std::vector<size_t> sliceSizes = compute_slice_sizes(dataShape);

    // Fill frames out-of-order to better test consistency.
    std::vector<size_t> framesToCheck{ 1, 0, 10, 100, 399 };
    std::vector<IndexNd<4>> indicesToCheck{
        IndexNd<4>{0, 0, 0, 0},
        IndexNd<4>{0, 0, 255, 0},
        IndexNd<4>{0, 128, 0, 128},
        IndexNd<4>{0, 128, 128, 128},
        IndexNd<4>{0, 255, 255, 0},
        IndexNd<4>{0, 255, 255, 255}
    };

    for (size_t frameIndex : framesToCheck)
    {
        for (auto indexNd : indicesToCheck)
        {
            indexNd[0] = frameIndex;

            uint8_t expectedValue = 1;
            for (size_t coord : indexNd)
                expectedValue = static_cast<uint8_t>((31 * expectedValue + coord) % 256);

            pNdarray->Write(index_to_vector<>(indexNd), expectedValue);
        }
    }
    pNdarray->FlushBuffer(true);

    SECTION("Data can be read back using the same instance.")
    {
        for (size_t frameIndex : framesToCheck)
        {
            bool allValuesEqual = true;
            for (auto indexNd : indicesToCheck)
            {
                indexNd[0] = frameIndex;

                uint8_t expectedValue = 1;
                for (size_t coord : indexNd)
                    expectedValue = static_cast<uint8_t>((31 * expectedValue + coord) % 256);

                // We expect the unfilled frames to be zeroed-out.
                if (std::find(framesToCheck.begin(), framesToCheck.end(), indexNd[0]) == framesToCheck.end())
                    expectedValue = 0;

                uint8_t actualValue = pNdarray->Read(index_to_vector(indexNd));
                if (expectedValue != actualValue)
                {
                    size_t indexFlat = flattenIndex(index_to_vector(indexNd), dataShape);
                    printf("Expected %u, got %u instead at index %zu. \n", expectedValue, actualValue, indexFlat);
                    allValuesEqual = false;
                    break;
                }
            }
            
            REQUIRE(allValuesEqual);
        }
    }
}


TEST_CASE("Basic functionality test.", "[smooth_3d_array_average]") {
    std::vector<double> data {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 0, 1, 2,

        3, 4, 5, 6,
        7, 8, 9, 0,
        1, 2, 3, 4,

        5, 6, 7, 8,
        9, 0, 1, 2,
        3, 4, 5, 6
    };

    IndexNd<3> dataShape{ 3, 3, 4 };
    std::vector<double> expectedResult {
        4.50, 5.00, 5.16, 5.25,
        4.00, 4.22, 4.11, 4.33,
        4.75, 4.83, 4.16, 4.25,

        4.66, 4.88, 4.77, 5.00,
        4.16, 4.29, 4.18, 4.50,
        4.50, 4.44, 3.77, 4.00,

        5.25, 5.33, 4.66, 4.75,
        4.33, 4.55, 4.44, 4.66,
        4.25, 4.33, 3.66, 3.75
    };

    std::vector<double> actualResult(expectedResult.size());
    smooth_3d_array_average(data.data(), IndexNd<3>{3, 3, 4}, 1, actualResult.data());

    for (size_t indexFlat = 0; indexFlat < 3 * 3 * 4; ++indexFlat)
    {
        IndexNd<3> indexNd{};
        unflattenIndex_fast(indexFlat, compute_slice_sizes_fast<3>(dataShape), indexNd);

        REQUIRE(actualResult[indexFlat] == Approx(expectedResult[indexFlat]).epsilon(0.1));
    }

}

TEST_CASE("Works on 2D volumes.", "[smooth_3d_array_average]") {
    std::vector<double> data {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 0, 1, 2,
    };

    IndexNd<3> dataShape{ 1, 3, 4 };
    std::vector<double> expectedResult {
        14.0/4, 24.0/6, 30.0/6, 22.0/4,
        23.0/6, 34.0/9, 33.0/9, 25.0/6,
        20.0/4, 28.0/6, 24.0/6, 18.0/4,
    };

    std::vector<double> actualResult(expectedResult.size());
    smooth_3d_array_average(data.data(), IndexNd<3>{1, 3, 4}, 1, actualResult.data());

    for (size_t indexFlat = 0; indexFlat < 1 * 3 * 4; ++indexFlat)
    {
        IndexNd<3> indexNd{};
        unflattenIndex_fast(indexFlat, compute_slice_sizes_fast<3>(dataShape), indexNd);
        REQUIRE(actualResult[indexFlat] == Approx(expectedResult[indexFlat]).epsilon(0.1));
    }
}

TEST_CASE("Works on small volumes with large kernel.", "[smooth_3d_array_average]") {
    std::vector<double> data {
        1, 2,
        5, 6,
    };

    IndexNd<3> dataShape{ 1, 2, 2 };
    std::vector<double> expectedResult {
        14.0/4, 14.0/4,
        14.0/4, 14.0/4
    };

    std::vector<double> actualResult(expectedResult.size());
    smooth_3d_array_average(data.data(), IndexNd<3>{1, 2, 2}, 3, actualResult.data());

    for (size_t indexFlat = 0; indexFlat < 1 * 2 * 2; ++indexFlat)
    {
        IndexNd<3> indexNd{};
        unflattenIndex_fast(indexFlat, compute_slice_sizes_fast<3>(dataShape), indexNd);
        REQUIRE(actualResult[indexFlat] == Approx(expectedResult[indexFlat]).epsilon(0.1));
    }
}