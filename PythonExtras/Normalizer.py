import pickle
from typing import *

import numpy as np

from PythonExtras import numpy_extras as npe


class Normalizer:
    """
    Normalizes input data to mean of zero and variance of one.
    Unlike scikit, supports N-dimensional arrays, with features on an arbitrary axis.
    """

    def __init__(self, minStd: float = 1e-3):
        self._axis = 0  # type: Union[None, int]
        self._hasFeatureDim = True
        self._featureNum = 0
        self._ndim = 0
        self._coefs = np.array([])
        self.isFit = False
        self._minStd = minStd
        self._dtype = None  # type: np.dtype

    def fit(self, data: np.ndarray, axis: Optional[int] = None):

        if np.issubdtype(data.dtype, np.integer):
            raise ValueError("Usage of Normalizer with integer types can lead to overflow.")

        self._dtype = data.dtype
        self._ndim = data.ndim

        if axis is not None:
            if axis < 0:
                axis += self._ndim
            self._featureNum = data.shape[axis]
        else:
            self._featureNum = 1
            self._hasFeatureDim = False

        # data = data.copy()
        self._axis = axis
        self._coefs = np.empty((self._featureNum, 2))

        for i, selector in self._enumerate_features():
            # Extract the feature.
            feature = data[selector]
            # Compute the moments.
            mean = np.mean(feature)
            std = np.std(feature)
            # Save the moments for future use
            self._coefs[i, :] = [mean, std]

        self.isFit = True

        return self

    def set_coefs(self, dtype: np.dtype, ndim: int, coefs: List[List[float]], axis: Optional[int] = None):
        if axis is not None:
            raise NotImplementedError()

        self._dtype = dtype
        self._ndim = ndim
        self._axis = axis
        self._featureNum = 1
        self._hasFeatureDim = False

        self._coefs = np.asarray(coefs, dtype=dtype)
        if self._coefs.shape != (1, 2):
            raise ValueError(f"Invalid coefs shape: f{self._coefs.shape}")

        self.isFit = True

    def scale_all(self, data: Iterable[np.ndarray], inPlace=False):
        for d in data:
            self.scale(d, inPlace=inPlace)

    def scale(self, data: np.ndarray, inPlace=False):

        if np.issubdtype(data.dtype, np.integer):
            raise ValueError("Usage of Normalizer with integer types can lead to overflow.")

        if self._dtype != data.dtype:
            raise ValueError("Normalizer has been 'fit()' with dtype {}, but dtype '{}' is given to 'scale()'."
                             .format(self._dtype, data.dtype))

        if not inPlace:
            data = data.copy()

        for i, selector in self._enumerate_features():
            data[selector] -= self._coefs[i, 0]
            if not self.is_feature_degenerate(i):  # Guard from division be zero or very small values.
                data[selector] /= self._coefs[i, 1]

        return data

    def fit_and_scale(self, data: npe.LargeArray, axis: Optional[int] = None, inPlace: bool = False):
        self.fit(data, axis)
        return self.scale(data, inPlace=inPlace)

    def fit_and_scale_batched(self, data: npe.LargeArray, axis: Optional[int] = None, inPlace: bool = True,
                              batchSizeFlat: int = int(1e9), doFit: bool = True, printFn: Optional[Callable] = print):
        if not inPlace:
            raise ValueError("Batched normalization by copy is not supported.")

        for i, (batchStart, batchEnd) in enumerate(npe.get_batch_indices(data.shape, data.dtype,
                                                                         batchSizeFlat=batchSizeFlat)):
            if printFn:
                printFn("Normalizing a batch of data.")
            if doFit and i == 0:
                self.fit(data[batchStart:batchEnd], axis=axis)

            data[batchStart:batchEnd] = self.scale(data[batchStart:batchEnd])

    def scale_batched(self, data: npe.LargeArray, inPlace: bool = True,
                      batchSizeFlat: int = int(1e9), printFn: Optional[Callable] = print):
        """
        This method is just a cleaner shortcut for 'fit_and_scale_batched'.
        """

        self.fit_and_scale_batched(data, axis=self._axis, inPlace=inPlace, batchSizeFlat=batchSizeFlat,
                                   doFit=False, printFn=printFn)

    def scale_back(self, data: np.ndarray):
        data = data.copy()
        for i, selector in self._enumerate_features():
            data[selector] = data[selector] * self._coefs[i, 1] + self._coefs[i, 0]

        return data

    def is_feature_degenerate(self, featureIndex):
        self._throw_if_not_fit()

        return self._coefs[featureIndex, 1] < self._minStd

    def zero_degenerate_features(self, data):
        self._throw_if_not_fit()

        degenerateFeatures = [i for i in range(data.shape[self._axis]) if self.is_feature_degenerate(i)]
        data[self._get_selector(degenerateFeatures, self._axis)] = 0
        return data

    def get_weights(self):
        return self._coefs.copy()

    def save(self, filepath: str):
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

    def _enumerate_features(self):
        if self._hasFeatureDim:
            for featureIndex in range(self._featureNum):
                yield featureIndex, self._get_selector(featureIndex, self._axis)
        else:
            yield 0, (Ellipsis, )  # Trivial selector to get all the data.

    def _get_selector(self, index, axis):
        return tuple((slice(None) if a != axis else index) for a in range(0, self._ndim))

    def _throw_if_not_fit(self):
        if not self.isFit:
            raise RuntimeError("Invalid operation: Normalizer hasn't been fit to any data yet.")

    @classmethod
    def load(cls, filepath: str) -> 'Normalizer':
        with open(filepath, 'rb') as file:
            return pickle.load(file)