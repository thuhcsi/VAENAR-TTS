import tensorflow as tf
from numpy import pi as PI
from typing import Tuple
from modules.flow import InvertibleLinearFlow, ActNormFlow, TransformerCoupling


class BasePrior(tf.keras.layers.Layer):
    """ P(z|x): prior that generate the latent variables conditioned on x
    """

    def __init__(self, channels, name='BasePrior', **kwargs):
        super(BasePrior, self).__init__(name=name, **kwargs)
        self.channels = channels

    def call(self, inputs, targets_lengths, condition_lengths, training=None
             ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        :param targets_lengths: [batch, ]
        :param inputs: condition_inputs
        :param condition_lengths:
        :param training: boolean
        :return: tensor1: outputs, tensor2: log_probabilities
        """
        raise NotImplementedError

    def _initial_sample(self, targets_lengths, temperature=1.0):
        """
        :param targets_lengths: [batch,]
        :param temperature: standard deviation
        :return: initial samples with shape [batch_size, length, channels],
                 log-probabilities: [batch, ]
        """
        batch_size = tf.shape(targets_lengths)[0]
        length = tf.cast(tf.math.reduce_max(targets_lengths), 'int32')
        epsilon = tf.random.normal([batch_size, length, self.channels],
                                   mean=0.0, stddev=temperature)
        logprobs = -0.5 * (tf.math.log(2. * PI) + epsilon ** 2)
        seq_mask = tf.expand_dims(
            tf.sequence_mask(
                targets_lengths, dtype=tf.float32), axis=-1)  # [batch, max_time, 1]
        logprobs = tf.reduce_sum(seq_mask * logprobs, axis=[1, 2])  # [batch, ]
        return epsilon, logprobs

    def log_probability(self, z, condition_inputs, z_lengths=None, condition_lengths=None
                        ) -> tf.Tensor:
        """
        compute the log-probability of given latent variables, first run through the flow
        inversely to get the initial sample, then compute the
        :param z: latent variables
        :param condition_inputs: condition inputs
        :param z_lengths:
        :param condition_lengths:
        :return: the log-probability
        """
        raise NotImplementedError

    def init(self, *inputs, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Initiate the weights according to the initial input data
        :param inputs:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    def sample(self, targets_lengths, n_samples, condition_inputs, condition_lengths=None
               ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        :param targets_lengths:
        :param n_samples:
        :param condition_inputs:
        :param condition_lengths:
        :return: tensor1: samples: [batch, n_samples, max_lengths, dim]
                 tensor2: log-probabilities: [batch, n_samples]
        """
        raise NotImplementedError


class TransformerPrior(BasePrior):
    def __init__(self, n_blk, channels, n_transformer_blk, attention_dim,
                 attention_heads, temperature, ffn_hidden, inverse=False,
                 name='GlowPrior', **kwargs):
        super(TransformerPrior, self).__init__(channels, name, **kwargs)
        self.glow = []
        orders = ['upper', 'lower']
        for i in range(n_blk):
            order = orders[i % 2]
            actnorm = ActNormFlow(channels, inverse, name='actnorm_{}'.format(i))
            linear = InvertibleLinearFlow(channels, inverse,
                                          name='invertible_linear_{}'.format(i))
            affine_coupling = TransformerCoupling(channels=channels, inverse=inverse,
                                                  nblk=n_transformer_blk,
                                                  attention_dim=attention_dim,
                                                  attention_heads=attention_heads,
                                                  temperature=temperature,
                                                  ffn_hidden=ffn_hidden,
                                                  order=order,
                                                  name='transformerCoupling{}'.format(i))
            self.glow.append((actnorm, linear, affine_coupling))

    def call(self, inputs, targets_lengths, condition_lengths, training=None, temperature=1.0
             ) -> Tuple[tf.Tensor, tf.Tensor]:
        # 1. get initial noise
        epsilon, logprobs = self._initial_sample(targets_lengths, temperature=temperature)
        z = epsilon
        for step in self.glow:
            actnorm, linear, affine_coupling = step
            z, logdet = actnorm(z, targets_lengths)
            logprobs -= logdet
            z, logdet = linear(z, targets_lengths)
            logprobs -= logdet
            z, logdet = affine_coupling(inputs=z, condition_inputs=inputs,
                                        inputs_lengths=targets_lengths,
                                        condition_lengths=condition_lengths,
                                        training=training)
            logprobs -= logdet
        return z, logprobs

    def log_probability(self, z, condition_inputs, z_lengths=None,
                        condition_lengths=None, training=None
                        ) -> tf.Tensor:
        """
        :param z: [batch, max_time, dim]
        :param condition_inputs:
        :param z_lengths:
        :param condition_lengths:
        :param training
        :return: log-probabilities of z, [batch]
        """
        print('tracing back at prior log-probability')
        epsilon = z
        batch_size = tf.shape(z)[0]
        max_time = tf.shape(z)[1]
        accum_logdet = tf.zeros([batch_size, ], dtype=tf.float32)
        for step in reversed(self.glow):
            actnorm, linear, affine_coupling = step
            epsilon, logdet = affine_coupling.bwd_pass(inputs=epsilon,
                                                       condition_inputs=condition_inputs,
                                                       inputs_lengths=z_lengths,
                                                       condition_lengths=condition_lengths,
                                                       training=training)
            accum_logdet += logdet
            epsilon, logdet = linear.bwd_pass(epsilon, z_lengths)
            accum_logdet += logdet
            epsilon, logdet = actnorm.bwd_pass(epsilon, z_lengths)
            accum_logdet += logdet
        logprobs = -0.5 * (tf.math.log(2. * PI) + epsilon ** 2)
        seq_mask = tf.expand_dims(
            tf.sequence_mask(z_lengths, max_time, dtype=tf.float32), axis=-1)  # [batch, max_time]
        logprobs = tf.reduce_sum(seq_mask * logprobs, axis=[1, 2])  # [batch, ]
        logprobs += accum_logdet
        return logprobs

    def sample(self, targets_lengths, n_samples, condition_inputs,
               condition_lengths=None, training=None, temperature=1.0):
        # 1. get initial noise
        max_inputs_len = tf.math.reduce_max(targets_lengths)
        targets_lengths = tf.tile(
            tf.expand_dims(targets_lengths, axis=-1), tf.constant([1, n_samples]))
        targets_lengths = tf.reshape(targets_lengths, [-1])
        epsilon, logprobs = self._initial_sample(targets_lengths, temperature=temperature)  # [batch*n_samples, ]
        # 2. expand condition inputs
        batch_size = tf.shape(condition_inputs)[0]
        max_cond_len = tf.shape(condition_inputs)[1]
        cond_dim = tf.shape(condition_inputs)[2]
        condition_inputs = tf.tile(
            tf.expand_dims(condition_inputs, axis=1),  # [batch, 1, max_time, dim]
            tf.constant([1, n_samples, 1, 1]))  # [batch, n_samples, max_time, dim]
        condition_inputs = tf.reshape(condition_inputs,
                                      [batch_size * n_samples, max_cond_len, cond_dim])
        # condition_inputs.set_shape([None, None, 2 * HPS.Encoder.Tacotron.lstm_hidden])
        if condition_lengths is not None:
            condition_lengths = tf.tile(tf.expand_dims(condition_lengths, axis=-1),
                                        tf.constant([1, n_samples]))
            condition_lengths = tf.reshape(condition_lengths, [-1])
        z = epsilon
        for step in self.glow:
            actnorm, linear, affine_coupling = step
            z, logdet = actnorm(z, targets_lengths)
            logprobs -= logdet
            z, logdet = linear(z, targets_lengths)
            logprobs -= logdet
            z, logdet = affine_coupling.fwd_pass(inputs=z, condition_inputs=condition_inputs,
                                                 inputs_lengths=targets_lengths,
                                                 condition_lengths=condition_lengths,
                                                 training=training)
            logprobs -= logdet
        z = tf.reshape(z, [batch_size, n_samples, max_inputs_len, self.channels])
        logprobs = tf.reshape(logprobs, [batch_size, n_samples])
        return z, logprobs

    def init(self, conditions, targets_lengths, condition_lengths, training=None):
        # 1. get initial noise
        epsilon, logprobs = self._initial_sample(targets_lengths)
        z = epsilon
        for step in self.glow:
            actnorm, linear, affine_coupling = step
            z, logdet = actnorm.init(z, targets_lengths)
            logprobs -= logdet
            z, logdet = linear(z, targets_lengths)
            logprobs -= logdet
            z, logdet = affine_coupling.init(inputs=z, condition_inputs=conditions,
                                             inputs_lengths=targets_lengths,
                                             condition_lengths=condition_lengths,
                                             training=training)
            logprobs -= logdet
        return z, logprobs