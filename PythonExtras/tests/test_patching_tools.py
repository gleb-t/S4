import unittest

import PythonExtras.patching_tools as patching_tools


class PatchingToolsTest(unittest.TestCase):


    def test_compute_prediction_domain(self):

        low, high = patching_tools.compute_prediction_domain(
            volumeShape=(7, 7, 7, 7),
            patchXSize=(2, 2, 2, 2),
            patchYSize=(2, 2, 2, 2),
            patchStride=(2, 2, 2, 2),
            patchInnerStride=(2, 2, 2, 2),
            predictionDelay=1
        )

        self.assertEqual(low, (3, 1, 1, 1))
        self.assertEqual(high, (7, 7, 7, 7))

        low, high = patching_tools.compute_prediction_domain(
            volumeShape=(100, 100, 100, 100),
            patchXSize=(5, 5, 5, 5),
            patchYSize=(1, 1, 1, 1),
            patchStride=(1, 1, 1, 1),
            patchInnerStride=(1, 1, 1, 1),
            predictionDelay=3
        )

        self.assertEqual(low, (7, 2, 2, 2))
        self.assertEqual(high, (100, 98, 98, 98))
