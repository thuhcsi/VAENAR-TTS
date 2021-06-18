import tensorflow as tf
from .utils import Conv1D


class LengthPredictor(tf.keras.layers.Layer):
    def __init__(self, n_conv, conv_filter, conv_kernel, drop_rate,
                 activation, bn_before_act, name='lengthPredictor'):
        super(LengthPredictor, self).__init__(name=name)
        self.conv_stack = []
        for i in range(n_conv):
            conv = Conv1D(filters=conv_filter, kernel_size=conv_kernel,
                          activation=activation, drop_rate=drop_rate,
                          bn_before_act=bn_before_act)
            self.conv_stack.append(conv)
        self.projection = tf.keras.layers.Dense(units=1)

    def call(self, inputs, input_lengths, training=None):
        conv_outs = inputs
        for conv in self.conv_stack:
            conv_outs = conv(conv_outs, training=training)
        proj_outs = self.projection(conv_outs)
        mask = tf.expand_dims(
            tf.sequence_mask(
                input_lengths, maxlen=tf.shape(inputs)[1], dtype=tf.float32),
            axis=-1)
        target_lengths = tf.reduce_sum(tf.exp(proj_outs) * mask, axis=[1, 2])
        return target_lengths


class DenseLengthPredictor(tf.keras.layers.Layer):
    def __init__(self, activation, name='lengthPredictor'):
        super(DenseLengthPredictor, self).__init__(name=name)
        self.projection = tf.keras.layers.Dense(units=1, activation=activation)

    def call(self, inputs, input_lengths, training=None):
        proj_outs = self.projection(inputs)
        mask = tf.expand_dims(
            tf.sequence_mask(
                input_lengths, maxlen=tf.shape(inputs)[1], dtype=tf.float32),
            axis=-1)
        target_lengths = tf.reduce_sum(tf.exp(proj_outs) * mask, axis=[1, 2])
        return target_lengths
