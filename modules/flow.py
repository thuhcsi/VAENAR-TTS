import tensorflow as tf
import numpy as np
from typing import Tuple
from modules.transform import TransformerTransform


class BaseFlow(tf.keras.layers.Layer):
    def __init__(self, inverse, name='BaseFlow', **kwargs):
        super(BaseFlow, self).__init__(name=name, **kwargs)
        self.inverse = inverse

    def _forward(self, *inputs, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """

        Args:
            *inputs: input [batch, *input_size]

        Returns: out: Tensor [batch, *input_size], logdet: Tensor [batch]
            out, the output of the flow
            logdet, the log determinant of :math:`\partial output / \partial input`
        """
        raise NotImplementedError

    def _backward(self, *inputs, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """

        Args:
            *input: input [batch, *input_size]

        Returns: out: Tensor [batch, *input_size], logdet: Tensor [batch]
            out, the output of the flow
            logdet, the log determinant of :math:`\partial output / \partial input`
        """
        raise NotImplementedError

    def call(self, *inputs, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """

        Args:
            *inputs: input [batch, *input_size]

        Returns: out: Tensor [batch, *input_size], logdet: Tensor [batch]
            out, the output of the flow
            logdet, the log determinant of :math:`\partial output / \partial input`
        """
        if self.inverse:
            return self._backward(*inputs, **kwargs)
        return self._forward(*inputs, **kwargs)

    def init(self, *inputs, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Initiate the weights according to the initial input data
        :param inputs:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def fwd_pass(self, inputs, *h, init=False, init_scale=tf.constant(1.0),
                 **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """

        Args:
            inputs: Tensor
                The random variable before flow
            h: list of object
                other conditional inputs
            init: bool
                perform initialization or not (default: False)
            init_scale: float
                initial scale (default: 1.0)

        Returns: y: Tensor, logdet: Tensor
            y, the random variable after flow
            logdet, the log determinant of :math:`\partial y / \partial x`
            Then the density :math:`\log(p(y)) = \log(p(x)) - logdet`

        """
        if self.inverse:
            if init:
                raise RuntimeError(
                    'inverse flow shold be initialized with backward pass')
            else:
                return self._backward(inputs, *h, **kwargs)
        else:
            if init:
                return self.init(inputs, *h, init_scale=init_scale, **kwargs)
            else:
                return self._forward(inputs, *h, **kwargs)

    def bwd_pass(self, inputs, *h, init=False, init_scale=tf.constant(1.0),
                 **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        :param inputs: the random variable after the flow
        :param h: other conditional inputs
        :param init: bool, whether perform initialization or not
        :param init_scale: float (default: 1.0)
        :param kwargs:
        :return: x: the random variable before the flow,
                 log_det: the log determinant of :math:`\partial x / \partial y`
            Then the density :math:`\log(p(y)) = \log(p(x)) + logdet`
        """
        if self.inverse:
            if init:
                return self.init(inputs, *h, init_scale=init_scale, **kwargs)
            else:
                return self._forward(inputs, *h, **kwargs)
        else:
            if init:
                raise RuntimeError(
                    'forward flow should be initialzed with forward pass')
            else:
                return self._backward(inputs, *h, **kwargs)


class InvertibleLinearFlow(BaseFlow):
    def __init__(self, channels, inverse, name='InvertibleLinearFlow', **kwargs):
        super(InvertibleLinearFlow, self).__init__(inverse, name, **kwargs)
        self.channels = channels
        w_init = np.linalg.qr(np.random.randn(channels, channels))[0].astype(np.float32)
        self.weight = tf.Variable(w_init, dtype=tf.float32, trainable=True, name='weight')

    def _forward(self, inputs, inputs_lengths=None, training=None
                 ) -> Tuple[tf.Tensor, tf.Tensor]:
        input_shape = tf.shape(inputs)
        outputs = tf.matmul(inputs, self.weight)
        logdet = tf.cast(
            tf.linalg.slogdet(
                tf.cast(self.weight, 'float64'))[1], 'float32')
        if inputs_lengths is None:
            logdet = tf.ones([input_shape[0], ]
                             ) * tf.cast(input_shape[1], tf.float32) * logdet
        else:
            logdet = tf.cast(inputs_lengths, tf.float32) * logdet
        return outputs, logdet

    def _backward(self, inputs, inputs_lengths=None, training=None
                  ) -> Tuple[tf.Tensor, tf.Tensor]:
        input_shape = tf.shape(inputs)
        outputs = tf.matmul(inputs, tf.linalg.inv(self.weight))
        logdet = tf.cast(
            tf.linalg.slogdet(
                tf.linalg.inv(
                    tf.cast(self.weight, 'float64')))[1], 'float32')
        if inputs_lengths is None:
            logdet = tf.ones([input_shape[0], ]
                             ) * tf.cast(input_shape[1], tf.float32) * logdet
        else:
            logdet = tf.cast(inputs_lengths, tf.float32) * logdet
        return outputs, logdet

    def init(self, inputs, inputs_lengths=None):
        return self._forward(inputs, inputs_lengths)


class ActNormFlow(BaseFlow):
    def __init__(self, channels, inverse, name='ActNormFlow', **kwargs):
        super(ActNormFlow, self).__init__(inverse, name, **kwargs)
        self.channels = channels
        self.log_scale = tf.Variable(
            tf.random.normal(shape=[self.channels, ], mean=0.0, stddev=0.05),
            trainable=True, name='log_scale')
        self.bias = tf.Variable(tf.zeros([self.channels, ]),
                                trainable=True, name='bias')

    def _forward(self, inputs, input_lengths=None, training=None) -> Tuple[tf.Tensor, tf.Tensor]:
        input_shape = tf.shape(inputs)
        outputs = inputs * tf.exp(self.log_scale) + self.bias
        logdet = tf.reduce_sum(self.log_scale)
        if input_lengths is None:
            logdet = tf.ones([input_shape[0], ]
                             ) * tf.cast(input_shape[1], tf.float32) * logdet
        else:
            logdet = tf.cast(input_lengths, tf.float32) * logdet
        return outputs, logdet

    def _backward(self, inputs, input_lengths=None, training=None, epsilon=tf.constant(1e-8)
                  ) -> Tuple[tf.Tensor, tf.Tensor]:
        input_shape = tf.shape(inputs)
        outputs = (inputs - self.bias) / (tf.exp(self.log_scale) + epsilon)
        logdet = -tf.reduce_sum(self.log_scale)
        if input_lengths is None:
            logdet = tf.ones([input_shape[0], ]
                             ) * tf.cast(input_shape[1], tf.float32) * logdet
        else:
            logdet = tf.cast(input_lengths, tf.float32) * logdet
        return outputs, logdet

    def init(self, inputs, input_lengths=None, init_scale=1.0, epsilon=tf.constant(1e-8)):
        _mean = tf.math.reduce_mean(
            tf.reshape(inputs, [-1, self.channels]), axis=0)
        _std = tf.math.reduce_std(
            tf.reshape(inputs, [-1, self.channels]), axis=0)
        self.log_scale.assign(tf.math.log(init_scale / (_std + epsilon)))
        self.bias.assign(-_mean / (_std + epsilon))
        return self._forward(inputs, input_lengths)


class TransformerCoupling(BaseFlow):
    def __init__(self, channels, inverse, nblk, attention_dim, attention_heads,
                 temperature, ffn_hidden, order='upper', name='affine_coupling', **kwargs):
        # assert channels % 2 == 0
        out_dim = channels // 2
        self.channels = channels
        super(TransformerCoupling, self).__init__(inverse, name, **kwargs)
        self.net = TransformerTransform(
            nblk=nblk, attention_dim=attention_dim, attention_heads=attention_heads,
            temperature=temperature, ffn_hidden=ffn_hidden, out_dim=out_dim)
        self.upper = (order == 'upper')

    @staticmethod
    def _split(inputs):
        return tf.split(inputs, num_or_size_splits=2, axis=-1)

    @staticmethod
    def _affine(inputs, scale, shift):
        return scale * inputs + shift

    @staticmethod
    def _inverse_affine(inputs, scale, shift, epsilon=tf.constant(1e-12)):
        return (inputs - shift) / (scale + epsilon)

    def _forward(self, inputs, condition_inputs,
                 inputs_lengths=None, condition_lengths=None, training=None
                 ) -> Tuple[tf.Tensor, tf.Tensor]:
        # assert tf.shape(inputs)[-1] == self.channels
        lower_pt, upper_pt = self._split(inputs)
        z, zp = (lower_pt, upper_pt) if self.upper else (upper_pt, lower_pt)
        log_scale, shift = self.net(z, condition_inputs, condition_lengths,
                                    inputs_lengths, training=training)
        scale = tf.math.sigmoid(log_scale + 2.0)
        zp = self._affine(zp, scale, shift)
        inputs_max_time = tf.shape(inputs)[1]
        mask = (tf.expand_dims(tf.sequence_mask(inputs_lengths, maxlen=inputs_max_time,
                                                dtype=tf.float32), axis=-1)
                if inputs_lengths is not None else tf.ones_like(log_scale))
        logdet = tf.reduce_sum(tf.math.log(scale) * mask, [1, 2])  # [batch, ]
        outputs = tf.concat([z, zp], axis=-1) if self.upper else tf.concat([zp, z], axis=-1)
        return outputs, logdet

    def _backward(self, inputs, condition_inputs,
                  inputs_lengths=None, condition_lengths=None, training=None
                  ) -> Tuple[tf.Tensor, tf.Tensor]:
        # assert tf.shape(inputs)[-1] == self.channels
        lower_pt, upper_pt = self._split(inputs)
        z, zp = (lower_pt, upper_pt) if self.upper else (upper_pt, lower_pt)
        log_scale, shift = self.net(z, condition_inputs, condition_lengths,
                                    inputs_lengths, training=training)
        scale = tf.math.sigmoid(log_scale + 2.0)
        zp = self._inverse_affine(zp, scale, shift)
        inputs_max_time = tf.shape(inputs)[1]
        mask = (tf.expand_dims(tf.sequence_mask(inputs_lengths, maxlen=inputs_max_time,
                                                dtype=tf.float32), axis=-1)
                if inputs_lengths is not None else tf.ones_like(log_scale))
        log_det = -tf.reduce_sum(tf.math.log(scale) * mask, [1, 2])  # [batch,]
        outputs = tf.concat([z, zp], axis=-1) if self.upper else tf.concat([zp, z], axis=-1)
        return outputs, log_det

    def init(self, inputs, condition_inputs, inputs_lengths=None,
             condition_lengths=None, training=None):
        return self._forward(
            inputs, condition_inputs, inputs_lengths, condition_lengths, training=training)
