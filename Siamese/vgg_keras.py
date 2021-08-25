from typing import List, Tuple, Optional

import numpy as np
import tensorflow as tf


def vgg_preprocess(inputsX: List[np.ndarray], inputsY: Optional[List[np.ndarray]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    from keras.applications.vgg16 import preprocess_input

    # Expected input: [Batch, Patch, T, Z, Y, X, Attr]
    # Drop T, Z and attr.
    inputsX = [x[:, :, 0, 0, :, :, 0] for x in inputsX]
    # Repeat the attribute axis to convert from "greyscale" to RGB expected by VGG.
    inputsX = [np.stack((x, x, x), axis=-1) for x in inputsX]

    # Scale our data to match the ImageNet mean/std.
    # Then, keras's VGG preprocessing will scale to whatever the model expects.
    cylinderMean, cylinderStd = 32.16, 17.25
    imagenetMean, imagenetStd = 0.449 * 255, 0.226 * 255
    inputsX = [(x - cylinderMean) * (imagenetStd / cylinderStd) + imagenetMean for x in inputsX]

    # Apply the keras's preprocessing function to convert to pretrained model's format.
    inputsX = [preprocess_input(x) for x in inputsX]

    return inputsX, inputsY