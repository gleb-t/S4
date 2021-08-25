from typing import List, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
from deprecation import deprecated

from PythonExtras import volume_tools
from PythonExtras.CppWrapper import CppWrapper
from PythonExtras.volume_tools import TTFFunc


class ImageRenderer:

    def __init__(self):
        self.imageCache = {}
        self.tfCache = None
        self.lastTfFunc = None

    def render_image(self, imageData: np.ndarray, imageHash: Optional[str], tfFunc: TTFFunc):
        """

        :param imageData:
        :param imageHash:  A unique identifier of this image data, or None when no caching is needed.
        :param tfFunc:
        :return:
        """
        # WE ASSUME HERE THAT TF DOESN'T CHANGE DYNAMICALLY. (same func == same TF)

        if tfFunc != self.lastTfFunc:
            self.tfCache = precompute_tf_uint8(tfFunc)
            self.lastTfFunc = tfFunc
            self.imageCache.clear()  # Reset the image cache, since the TF has changed.

        if imageHash not in self.imageCache:
            if len(imageData.shape) == 2:
                imageData = imageData[np.newaxis, ...]

            # Drop the temporal and Z dimensions to get a 2D+C image.
            image = render_2d_volume_to_rgba(imageData, self.tfCache)[0, 0]

            if imageHash is not None:
                self.imageCache[imageHash] = image

            return image
        else:
            return self.imageCache[imageHash]


class Volume2DRenderer(ImageRenderer):

    def render_volume(self, volumeData: np.ndarray, tfFunc: TTFFunc):
        if type(tfFunc) == str:
            tfFunc = plt.get_cmap(tfFunc)

        assert volumeData.dtype == np.uint8

        if len(volumeData.shape) == 3:
            volumeData = volumeData.reshape((1,) + volumeData.shape)

        # Apply the TF to the volume.
        images = np.empty(volumeData.shape + (4,), dtype=np.uint8)
        for f in range(volumeData.shape[0]):
            images[f] = self.render_image(volumeData[f], None, tfFunc)

        return images


def render_2d_volume_files_to_rgba(inputDatPath: str,
                                   transferFunction: Union[volume_tools.TTFFunc, str],
                                   timeSlice: slice = None,
                                   timesteps: List[int] = None,
                                   printFn=None) -> np.ndarray:

    data = volume_tools.load_volume_data_from_dat(inputDatPath)
    if timeSlice:
        data = data[timeSlice, ...]
    elif timesteps:
        data = data[timesteps, ...]

    return render_2d_volume_to_rgba(data, transferFunction, printFn)


def precompute_tf_uint8(transferFunction: volume_tools.TTFFunc):
    tfArray = np.empty((256, 4), dtype=np.uint8)
    for val in range(256):
        tfArray[val, :] = [int(255 * x) for x in transferFunction(val)]

    return tfArray


@deprecated("Use Volume2DRenderer instead.")
def render_2d_volume_to_rgba(data: np.ndarray,
                             transferFunction: Union[volume_tools.TTFFunc, np.ndarray, str],
                             printFn=None) -> np.ndarray:
    """

    :param data:
    :param transferFunction: Callable TFs should be [0, 255]=>[0, 1]; precomputed [0, 255]=>[0, 255].
    :param printFn:
    :return:
    """

    if type(transferFunction) == str:
        transferFunction = plt.get_cmap(transferFunction)
    if callable(transferFunction):
        tfArray = precompute_tf_uint8(transferFunction)
    else:
        tfArray = transferFunction
        assert tfArray.shape == (256, 4)
        assert tfArray.dtype == np.uint8

    assert data.dtype == np.uint8

    if len(data.shape) == 3:
        data = data[np.newaxis, ...]

    # Apply the TF to the volume.
    images = np.empty(data.shape + (4,), dtype=np.uint8)
    CppWrapper.apply_tf_to_volume_uint8(data, tfArray, images)

    return images


