import math
from typing import *

import keras
from keras.layers.convolutional import _Conv
from keras import backend as kb
import tensorflow as tf


class L1Layer(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(L1Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        assert len(input_shape) == 2

        # Don't have any trainable parameters.

        super(L1Layer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, list)
        assert len(inputs) == 2

        left, right = inputs

        return [kb.abs(left - right)]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        assert len(input_shape) == 2

        shapeLeft, shapeRight = input_shape

        if shapeLeft != shapeRight:
            raise ValueError("Incompatible shapes: '{}' and '{}'.".format(shapeLeft, shapeRight))

        return shapeLeft


class MulConstLayer(keras.layers.Layer):

    def __init__(self, const, **kwargs):
        super(MulConstLayer, self).__init__(**kwargs)

        self.const = const

    def get_config(self):
        config = {
            'const': self.const
        }
        baseConfig = super(MulConstLayer, self).get_config()
        return {**baseConfig, **config}

    def build(self, input_shape):
        assert not isinstance(input_shape, list)

        # Don't have any trainable parameters.

        super(MulConstLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        assert not isinstance(inputs, list)

        return inputs * self.const

    def compute_output_shape(self, input_shape):
        assert not isinstance(input_shape, list)

        return input_shape


class SplitLayer(keras.layers.Layer):

    def __init__(self, axis: int, length: int, **kwargs):
        super(SplitLayer, self).__init__(**kwargs)

        self.axis = axis
        self.length = length

    def get_config(self):
        config = {
            'axis': self.axis,
            'length': self.length
        }
        baseConfig = super(SplitLayer, self).get_config()
        return {**baseConfig, **config}

    def build(self, input_shape):
        assert not isinstance(input_shape, list)

        self.ndim = len(input_shape)

        # Don't have any trainable parameters.
        super(SplitLayer, self).build(input_shape)

    def call(self, input, **kwargs):
        assert not isinstance(input, list)

        # Separate each slice along the target dimension into its own tensor.
        outputs = []
        for iSlice in range(self.length):
            selector = tuple(iSlice if dim == self.axis else slice(None)
                             for dim in range(self.ndim))
            outputs.append(input[selector])

        return outputs

    def compute_output_shape(self, input_shape):
        assert not isinstance(input_shape, list)

        sliceShape = tuple(x for dim, x in enumerate(input_shape) if dim != self.axis)

        return [sliceShape] * self.length


class Conv4d(keras.layers.Layer):
    """
    A keras layer wrapper around our tensorflow conv4d implementation.
    """
    def __init__(self, filters: int, kernelSize: Tuple[int, ...], strides=(1, 1, 1, 1), padding='valid',
                 activation=None, kernel_regularizer=None, **kwargs):
        self.filters = filters
        self.kernelSize = tuple(kernelSize)  # Keras deserializes tuples as lists, make sure to convert.
        self.strides = tuple(strides)
        self.padding=padding
        self.activation = keras.activations.get(activation)
        self.kernelReg = keras.regularizers.get(kernel_regularizer)

        self.kernel = None  # type: Optional[tf.Variable]
        self.bias = None  # type: Optional[tf.Variable]

        super(Conv4d, self).__init__(**kwargs)

    def build(self, input_shape):
        # Using channels last.

        ndimSpatial = len(self.kernelSize)
        ndimAll = len(input_shape)
        assert ndimAll == ndimSpatial + 2  # Spatial dimensions + batch size + channels.

        # (spatial1, spatial2,...., spatialN, input_channels, output_channels)
        kernelShape = self.kernelSize + (input_shape[-1], self.filters)
        # One bias term per output dimensions. Other dimensions left at '1' to broadcast.
        biasShape = tuple(1 for _ in range(ndimAll-1)) + (self.filters,)

        self.kernel = self.add_weight(name='kernel',
                                      shape=kernelShape,
                                      initializer='glorot_uniform',
                                      regularizer=self.kernelReg,
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=biasShape,
                                    initializer='zeros',
                                    trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        from AttentionFlow.tf_tools import conv4d_op
        # TF expec
        results = conv4d_op(inputs, self.kernel, (1, *self.strides, 1), self.padding.upper())
        results = results + self.bias

        if self.activation:
            results = self.activation(results)

        return results

    def compute_output_shape(self, inputShape: Tuple[int, ...]):
        sizes = inputShape[1:-1]

        if self.padding == 'valid':
            sizes = [int(math.ceil((inSize - kSize + 1) / stride))
                     for inSize, kSize, stride in zip(sizes, self.kernelSize, self.strides)]
        else:
            raise ValueError('Unsupported padding type: {}'.format(self.padding))

        return (inputShape[0], *sizes, self.filters)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernelSize': self.kernelSize,
            'strides': self.strides,
            'padding': self.padding,
            'activation': keras.activations.serialize(self.activation),
            'kernel_regularizer': keras.regularizers.serialize(self.kernelReg),
        }
        baseConfig = super(Conv4d, self).get_config()
        return {**baseConfig, **config}


class VggAdapterLayer(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(VggAdapterLayer, self).__init__(**kwargs)

        self.vggInputShape = (224, 224)

    def build(self, input_shape):
        assert not isinstance(input_shape, list)

        # Don't have any trainable parameters.
        super(VggAdapterLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        assert not isinstance(x, list)

        # Had issues with keras.backend.repeat_elements.
        # encoder.add(keras.layers.Lambda(lambda x: keras.backend.repeat_elements(x, 3, axis=3)))

        # Resize the image, since we'll used a pretrained VGG model. Do it as part of the model to maintain the interface.
        x = tf.image.resize_images(
            x,
            self.vggInputShape,
            method=tf.image.ResizeMethod.BILINEAR,
            # https://hackernoon.com/how-tensorflows-tf-image-resize-stole-60-days-of-my-life-aba5eb093f35
            align_corners=True,
            preserve_aspect_ratio=False
        )

        return x

    def compute_output_shape(self, input_shape):
        assert not isinstance(input_shape, list)

        return (input_shape[0], *self.vggInputShape, 3)


CUSTOM_KERAS_LAYERS = {
    'L1Layer': L1Layer,
    'MulConstLayer': MulConstLayer,
    'SplitLayer': SplitLayer,
    'Conv4d': Conv4d,
    'VggAdapterLayer': VggAdapterLayer
}