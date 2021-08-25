import os
import tempfile
from typing import Union, List, Tuple

import unittest
import numpy as np

from PythonExtras.BufferedNdArray import BufferedNdArray
import PythonExtras.numpy_extras as npe


class BufferedNdArrayTest(unittest.TestCase):
    """
    Some of the tests are implemented in C++.
    """

    def setUp(self):
        super().setUp()

        # Default test settings.
        self.shape = (64, 1, 256, 256)
        self.bufferSize = int(npe.multiply(self.shape) / 64 * 2) * 4
        tmpFileHandle, self.filepath = tempfile.mkstemp()
        os.fdopen(tmpFileHandle, 'w').close()  # Close the file handle immediately.

    def tearDown(self):
        super().tearDown()

        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def test_construction_destruction(self):

        if os.path.exists(self.filepath):
            os.unlink(self.filepath)

        with BufferedNdArray(self.filepath, BufferedNdArray.FileMode.rewrite, (10, 128, 128, 128),
                             np.uint8, int(1e5)) as array:
            pass

        self.assertTrue(os.path.exists(self.filepath))

    def test_mixed_reads_writes(self):
        sliceA = np.tile(np.arange(0, 256, dtype=np.float32), (256, 1)).copy()  # type: np.ndarray
        sliceB = (sliceA.transpose() * 2).copy()
        sliceC = np.ones((1, 256, 256), dtype=np.float32)  # A 3D slice.

        assert sliceA.shape == sliceB.shape == (256, 256)

        with BufferedNdArray(self.filepath, BufferedNdArray.FileMode.rewrite,
                             self.shape, np.float32, self.bufferSize) as data:
            data.fill_box(0, (0,) * data.ndim, data.shape)

            data[0, 0] = sliceA
            data[1, 0] = sliceB

            data[20, 0] = sliceA

            self.assertTrue(np.all(np.equal(data[0, 0], sliceA)))
            self.assertTrue(np.all(np.equal(data[1, 0], sliceB)))

            # Also test "uneven" slice dimensions, e.g. 3D slice of a 4D array. Had a bug here.
            # data[30] = sliceC
            # self.assertTrue(np.all(np.equal(data[30], sliceC)))

            data[62, 0] = sliceB
            data[63, 0] = sliceA

            data[0, 0, 0, 0] = 15.5
            data[63, 0, 255, 255] = 16.6

            data[1, 0, 100, 100] = 17.7
            data[25, 0, 150, 150] = 17.7

        with BufferedNdArray(self.filepath, BufferedNdArray.FileMode.readonly,
                             self.shape, np.float32, self.bufferSize) as data:
            self.assertTrue(np.all(np.equal(data[20, 0], sliceA)))

            self.assertTrue(np.all(np.equal(data[62, 0], sliceB)))

            self.assertAlmostEqual(data[0, 0, 0, 0], 15.5, places=4)  # Why did this suddenly stop working?
            self.assertAlmostEqual(data[63, 0, 255, 255], 16.6, places=4)

            self.assertAlmostEqual(data[1, 0, 100, 100], 17.7, places=4)
            self.assertAlmostEqual(data[25, 0, 150, 150], 17.7, places=4)

        with BufferedNdArray(self.filepath, BufferedNdArray.FileMode.update,
                             self.shape, np.float32, self.bufferSize) as data:

            with self.assertWarns(BufferedNdArray.NoncontinuousArray):
                data[0, 0] = sliceB.transpose()  # Give a 'view' which is not a cont. memory buffer.

            self.assertTrue(np.all(np.equal(data[0, 0], sliceB.transpose())))

    def test_slice_writes(self):

        shape = (10, 11, 12, 13, 14)

        # Test direct and buffer-mediated slice writes.
        for isDirect in (False, True):

            with BufferedNdArray(self.filepath, BufferedNdArray.FileMode.rewrite, shape, np.float32, int(1e4)) as array:
                array.set_direct_mode(isDirect)

                # Also do a few manual test with different slice sizes.
                array[9, 9, 9, ...] = (np.ones((13, 14), dtype=np.float32) * 14).copy()
                array[9, 9, 8, ...] = (np.ones((13, 14), dtype=np.float32) * 17).copy()
                array[6, 6, 6, 6, ...] = (np.ones((14,), dtype=np.float32) * 19).copy()
                array[6, 6, 6, 5, ...] = (np.ones((14,), dtype=np.float32) * 23).copy()
                array[6, 6, 6, 7, ...] = (np.ones((14,), dtype=np.float32) * 29).copy()

            with BufferedNdArray(self.filepath, BufferedNdArray.FileMode.readonly, shape, np.float32, int(1e4)) as array:
                self.assertTrue(np.all(np.equal(array[9, 9, 9], 14)))
                self.assertTrue(np.all(np.equal(array[9, 9, 8], 17)))
                self.assertTrue(np.all(np.equal(array[6, 6, 6, 6, ...], 19)))
                self.assertTrue(np.all(np.equal(array[6, 6, 6, 5, ...], 23)))
                self.assertTrue(np.all(np.equal(array[6, 6, 6, 7, ...], 29)))

            # Larger buffer to fit the bigger slices.
            with BufferedNdArray(self.filepath, BufferedNdArray.FileMode.update, shape, np.float32, int(1e6)) as array:
                array.set_direct_mode(isDirect)

                array[5, 5, ...] = (np.ones((12, 13, 14), dtype=np.float32) * 31).copy()
                array[5, 4, ...] = (np.ones((12, 13, 14), dtype=np.float32) * 37).copy()
                array[5, 6, ...] = (np.ones((12, 13, 14), dtype=np.float32) * 41).copy()

            with BufferedNdArray(self.filepath, BufferedNdArray.FileMode.readonly, shape, np.float32, int(1e6)) as array:
                self.assertTrue(np.all(np.equal(array[5, 5, ...], 31)))
                self.assertTrue(np.all(np.equal(array[5, 4, ...], 37)))
                self.assertTrue(np.all(np.equal(array[5, 6, ...], 41)))

    def test_query_type_is_recognized(self):

        shape = (8, 9, 10, 11, 12)

        ctor = BufferedNdArray.Query
        self.assertEqual(ctor(1, shape).type, BufferedNdArray.Query.Type.slice)
        self.assertEqual(ctor((1, 2, 3, 4, 5), shape).type, BufferedNdArray.Query.Type.cell)
        self.assertEqual(ctor((1, 2, 3, Ellipsis), shape).type, BufferedNdArray.Query.Type.slice)
        self.assertEqual(ctor((1, 2), shape).type, BufferedNdArray.Query.Type.slice)
        self.assertEqual(ctor((1, 2, None, None), shape).type, BufferedNdArray.Query.Type.slice)
        self.assertEqual(ctor((Ellipsis, 1, 2, None), shape).type, BufferedNdArray.Query.Type.slab)
        self.assertEqual(ctor((1, Ellipsis, 3, None), shape).type, BufferedNdArray.Query.Type.slab)
        self.assertEqual(ctor((slice(1, 2)), shape).type, BufferedNdArray.Query.Type.slab_outer)
        self.assertEqual(ctor((1, 2, slice(1, 2), None), shape).type, BufferedNdArray.Query.Type.slab)
        self.assertEqual(ctor((slice(1, 2), 2, slice(1, 2),), shape).type, BufferedNdArray.Query.Type.slab)

    def test_read_outer_slabs(self):

        shape = (8, 9, 10, 11, 12)
        with BufferedNdArray(self.filepath, BufferedNdArray.FileMode.rewrite, shape, np.float32, int(1e8)) as array:

            # Write the data into the array.
            data = np.arange(0, npe.multiply(shape), dtype=np.float32).reshape(shape)
            for i in range(shape[0]):
                array[i] = data[i]

            # Now try reading it out.
            np.testing.assert_equal(array[1:3], data[1:3])
            np.testing.assert_equal(array[2:5], data[2:5])
            np.testing.assert_equal(array[1:8], data[1:8])
            np.testing.assert_equal(array[7:8], data[7:8])

    def test_write_outer_slabs(self):

        shape = (8, 9, 10, 11, 12)
        # Write the data into the array.
        with BufferedNdArray(self.filepath, BufferedNdArray.FileMode.rewrite, shape, np.float32, int(1e6)) as array:
            # Also do a few manual test with different slice sizes.
            array[0:1, ...] = (np.ones((1,) + shape[1:], dtype=np.float32) * 13).copy()
            array[1:3, ...] = (np.ones((2,) + shape[1:], dtype=np.float32) * 14).copy()
            array[3:5, None, None, None, None] = (np.ones((2,) + shape[1:], dtype=np.float32) * 17).copy()
            array[4:8, :, :, :, :] = (np.ones((4,) + shape[1:], dtype=np.float32) * 19).copy()

        with BufferedNdArray(self.filepath, BufferedNdArray.FileMode.readonly, shape, np.float32, int(1e6)) as array:
            # Now try reading it out.
            np.testing.assert_array_equal(array[0:1, ...], 13)
            np.testing.assert_array_equal(array[1:3, ...], 14)
            np.testing.assert_array_equal(array[3:4, ...], 17)
            np.testing.assert_array_equal(array[4:8, ...], 19)


    def test_read_slabs(self):

        shape = (8, 9, 10, 11, 12)
        with BufferedNdArray(self.filepath, BufferedNdArray.FileMode.rewrite, shape, np.float32, int(1e8)) as array:

            # Write the data into the array.
            data = np.arange(0, npe.multiply(shape), dtype=np.float32).reshape(shape)
            for i in range(shape[0]):
                array[i] = data[i]

            # Now try reading it out.
            np.testing.assert_equal(array[1], data[1])
            np.testing.assert_equal(array[1, 3], data[1, 3])
            np.testing.assert_equal(array[3, 2:5], data[3, 2:5])
            np.testing.assert_equal(array[3, 2:5, 3], data[3, 2:5, 3])
            np.testing.assert_equal(array[1, :, 3, ...], data[1, :, 3, ...])
            np.testing.assert_equal(array[1, ..., :], data[1, ..., :])

    def test_write_read_full(self):
        shape = (8, 9, 10, 11, 12)
        with BufferedNdArray(self.filepath, BufferedNdArray.FileMode.rewrite, shape, np.float32, int(1e8)) as array:

            # Write the data into the array.
            data = np.arange(0, npe.multiply(shape), dtype=np.float32).reshape(shape)
            array[...] = data

            np.testing.assert_equal(array[...], data)

    def test_fill_box(self):
        shape = (5, 6, 10, 6)
        with BufferedNdArray(self.filepath, BufferedNdArray.FileMode.rewrite, shape, np.float32, int(1e8)) as array:
            array.fill_box(0.01, (0, 0, 0, 0), (5, 6, 5, 6))

            self.assertAlmostEqual(array[0, 0, 0, 0], 0.01)
            self.assertAlmostEqual(array[2, 2, 2, 2], 0.01)
            self.assertAlmostEqual(array[4, 5, 4, 5], 0.01)
            self.assertAlmostEqual(array[4, 0, 4, 5], 0.01)
            self.assertAlmostEqual(array[4, 5, 5, 5], 0)
            self.assertAlmostEqual(array[4, 5, 9, 5], 0)
            self.assertAlmostEqual(array[4, 5, 9, 5], 0)