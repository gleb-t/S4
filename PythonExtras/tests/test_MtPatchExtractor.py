from typing import Union, List, Tuple

import unittest
import tempfile
import os
import shutil
import h5py
import numpy as np

import PythonExtras.numpy_extras as npe
import PythonExtras.patching_tools as patching_tools
import PythonExtras.file_tools as file_tools
from PythonExtras.BufferedNdArray import BufferedNdArray
from PythonExtras.MtPatchExtractor import MtPatchExtractor


class MtPatchExtractorTest(unittest.TestCase):
    """
    Blatant copy-paste from the other patching test!

    Tests for the new patch-to-patch patching class.

    Because it's so cumbersome to create manual test cases (and hard to edit them later),
    the testing is performed against the single-threaded implementation, which is both covered
    by manual tests and has been in production for some time.
    """

    def setUp(self):
        super().setUp()

        # self.data = np.arange(0, 8 * 7 * 9 * 10, dtype=np.uint8).reshape((8, 7, 9, 10))
        # We have to use random values, otherwise patches repeat (due to overflow of uint8),
        # and it becomes hard to check the result for uniqueness.
        self.data = np.random.randint(0, 255, (8, 7, 9, 10), dtype=np.uint8)
        self.dataSizeFlat = npe.multiply(self.data.shape)  # type: int

        self.patchSizeOld = (4, 2, 2, 2)
        self.patchXSize = (self.patchSizeOld[0] - 1,) + self.patchSizeOld[1:]
        self.patchYSize = (1, 1, 1, 1)

        self.patchStride = (1, 1, 1, 1)
        self.patchedAxes = (0, 1, 2, 3)
        self.patchInnerStride = (2, 1, 1, 1)
        self.lastFrameGap = 2

        self.patchNumber = patching_tools.compute_patch_number_old(self.data.shape, self.patchedAxes, self.patchSizeOld,
                                                                   self.patchStride, patchInnerStride=self.patchInnerStride,
                                                                   lastFrameGap=self.lastFrameGap)

        self.patchNumberFlat = npe.multiply(self.patchNumber)
        self.setUpBuffers(self.patchNumberFlat)

        self.dataStartFlat = 0
        self.dataEndFlat = self.dataSizeFlat

        self.batchStartIndex = 0
        self.batchSize = self.patchNumberFlat

        self.patchExtent = [(self.patchSizeOld[dim] - 1) * self.patchInnerStride[dim] + 1 for dim in
                            range(len(self.patchSizeOld))]
        self.patchExtent[0] += self.lastFrameGap - self.patchInnerStride[0]

        lastPatchLower = patching_tools.patch_index_to_data_index_old(self.patchNumberFlat - 1, self.data.shape, (0, 1, 2, 3),
                                                                      self.patchSizeOld, self.patchStride)
        lastPatchYNd = np.asarray(lastPatchLower) + np.asarray(self.patchExtent)
        lastPatchYNd[0] -= 1

        self.lastPatchYFlat = npe.flatten_index(lastPatchYNd, self.data.shape)

        # load the data into a buffered array and build a patch extractor.
        self.tempDir = tempfile.mkdtemp()
        self.tempFile = os.path.join(self.tempDir, 'temp.raw')
        self.bufferedData = BufferedNdArray(self.tempFile, BufferedNdArray.FileMode.rewrite, self.data.shape,
                                            self.data.dtype, self.dataSizeFlat)
        for f in range(self.data.shape[0]):
            self.bufferedData[f] = self.data[f]

    def tearDown(self):
        super().tearDown()

        self.patchExtractor.destruct()

        self.bufferedData.destruct()
        shutil.rmtree(self.tempDir)

    def setUpBuffers(self, batchSize):
        self.batchX = np.empty((batchSize,) + self.patchXSize, dtype=self.data.dtype)
        self.batchY = np.empty((batchSize, 1), dtype=self.data.dtype)
        self.batchIndices = np.empty((batchSize, self.data.ndim), dtype=np.uint64)

    def test_all_in_one_batch(self):

        self.patchExtractor = MtPatchExtractor(self.bufferedData, self.patchXSize, self.patchYSize,
                                               self.patchStride, self.patchInnerStride, self.lastFrameGap,
                                               False, 0, 0, 1.0, 1.0, 1.0, self.dataSizeFlat)

        self.assertIdenticalToReference(batchSize=self.patchNumberFlat)

    def test_only_first_patch_fits(self):
        firstPatchEnd = npe.flatten_index(tuple(np.array(self.patchExtent) - 1), self.data.shape)

        self.patchExtractor = MtPatchExtractor(self.bufferedData, self.patchXSize, self.patchYSize,
                                               self.patchStride, self.patchInnerStride, self.lastFrameGap,
                                               False, 0, 0, 1.0, 1.0, 1.0, firstPatchEnd + 1)

        self.assertIdenticalToReference(dataEndFlat=firstPatchEnd + 1)

    def assertIdenticalToReference(self, data: np.ndarray = None, dataShape: Tuple = None, dataStartFlat: int = None,
                                   dataEndFlat: int = None, outputX: np.ndarray = None, outputY: np.ndarray = None,
                                   outputIndices: np.ndarray = None, patchSize: Tuple = None, patchStride: Tuple = None,
                                   batchStartIndex: int = None, batchSize: int = None,
                                   patchInnerStride: Tuple= None,
                                   lastFrameGap: int = None,
                                   undersamplingProb: float = None,
                                   skipEmptyPatches: bool = None,
                                   emptyValue: int = None):
        """
        Runs the reference single-threaded and the target multithreaded implementations
        and compares their results.
        Arguments can be provided to override the default values.
        """

        outputX = outputX if outputX is not None else self.batchX
        outputY = outputY if outputY is not None else self.batchY
        outputIndices = outputIndices if outputIndices is not None else self.batchIndices

        outputX[...] = 0
        outputY[...] = 0
        outputIndices[...] = 0

        # Run the reference single-threaded implementation.
        patchesExtractedRef, nextPatchIndexRef, inputEndReachedRef = \
            patching_tools.extract_patched_training_data_without_empty_4d(
                data if data is not None else self.data,
                dataShape or self.data.shape,
                dataStartFlat or self.dataStartFlat,
                dataEndFlat or self.dataEndFlat,
                outputX,
                outputY,
                outputIndices,
                patchSize or self.patchSizeOld,
                patchStride or self.patchStride,
                batchStartIndex or self.batchStartIndex,
                batchSize or self.batchSize,
                patchInnerStride=patchInnerStride or self.patchInnerStride,
                lastFrameGap=lastFrameGap or self.lastFrameGap,
                skipEmptyPatches=skipEmptyPatches or False,
                emptyValue=emptyValue or 0
            )

        self.referenceX = outputX.copy()
        self.referenceY = outputY.copy()
        self.referenceIndices = outputIndices.copy()

        outputX[...] = 0
        outputY[...] = 0
        outputIndices[...] = 0

        # Run the test target.
        patchesExtracted = \
            self.patchExtractor.extract_next_batch(self.batchSize, outputX, outputY, outputIndices)

        self.assertEqual(patchesExtractedRef, patchesExtracted)

        # The multithreaded function output can be shuffled, so don't compare directly,
        # but check that every patch is present in the reference output.
        # Since we check that the same number of patches is extracted and
        # the input data is random, we only need to check one way.
        for i in range(patchesExtracted):
            foundMatch = False
            for j in range(patchesExtracted):
                if np.all(np.equal(self.referenceX[i, ...], outputX[j, ...])):
                    np.testing.assert_array_equal(self.referenceY[i, ...], outputY[j, ...])
                    np.testing.assert_array_equal(self.referenceIndices[i, ...], outputIndices[j, ...])

                    foundMatch = True
                    break

            if not foundMatch:
                pass

            self.assertTrue(foundMatch)

        return patchesExtracted
