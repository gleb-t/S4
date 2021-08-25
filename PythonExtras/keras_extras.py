from typing import Optional

import keras


def assert_gpu_is_available():
    import tensorflow as tf
    if not tf.test.is_gpu_available(cuda_only=True):
        raise RuntimeError("No CUDA GPU available.")


def get_layer_recursive(model: keras.models.Model, name: str) -> Optional[keras.layers.Layer]:

    for layer in model.layers:
        if layer.name == name:
            return layer
        if isinstance(layer, keras.models.Model):
            result = get_layer_recursive(layer, name)
            if result is not None:
                return result

    return None
