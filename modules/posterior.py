import tensorflow as tf
import numpy as np

from typing import Tuple

from modules.utils import PostNet, CBHGLayer, PreNet, PositionalEncoding
from modules.attention import BahdanauAttention, CrossAttentionBLK


class BasePosterior(tf.keras.layers.Layer):
    """Encode the target sequence into latent distributions"""

    def __init__(self, name='Posterior', **kwargs):
        super(BasePosterior, self).__init__(name=name, **kwargs)

    def call(self, inputs, src_enc, src_lengths=None, target_lengths=None
             ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        raise NotImplementedError

    @staticmethod
    def reparameterize(mu, logvar, nsamples=tf.constant(1), random=tf.constant(True)):
        """
        :param mu: [batch, max_time, dim]
        :param logvar: [batch, max_time, dim]
        :param nsamples: int
        :param random: whether sample from N(0, 1) or just use zeros
        :return: samples, noises, [batch, nsamples, max_time, dim]
        """
        print('tracing back at posterior reparameterize')
        batch = tf.shape(mu)[0]
        max_time = tf.shape(mu)[1]
        dim = tf.shape(mu)[2]
        std = tf.math.exp(0.5 * logvar)
        if random:
            eps = tf.random.normal([batch, nsamples, max_time, dim])
        else:
            eps = tf.zeros([batch, nsamples, max_time, dim])
        samples = eps * tf.expand_dims(std, axis=1) + tf.expand_dims(mu, axis=1)
        return samples, eps

    @staticmethod
    def log_probability(mu, logvar, z=None, eps=None, seq_lengths=None, epsilon=tf.constant(1e-8)):
        """
        :param mu: [batch, max_time, dim]
        :param logvar: [batch, max_time, dim]
        :param z: [batch, nsamples, max_time, dim]
        :param eps: [batch, nsamples, max_time, dim]
        :param seq_lengths: [batch, ]
        :param epsilon: small float number to avoid overflow
        :return: log probabilities, [batch, nsamples]
        """
        print('tracing back at posterior log-probability')
        batch = tf.shape(mu)[0]
        max_time = tf.shape(mu)[1]
        dim = tf.shape(mu)[2]
        std = tf.math.exp(0.5 * logvar)
        normalized_samples = (eps if eps is not None
                              else (z - tf.expand_dims(mu, axis=1))
                                   / (tf.expand_dims(std, axis=1) + epsilon))
        expanded_logvar = tf.expand_dims(logvar, axis=1)
        # time_level_log_probs [batch, nsamples, max_time]
        time_level_log_probs = -0.5 * (
                tf.cast(dim, tf.float32) * tf.math.log(2 * np.pi)
                + tf.reduce_sum(expanded_logvar + normalized_samples ** 2.,
                                axis=3))
        seq_mask = (tf.sequence_mask(seq_lengths, maxlen=max_time, dtype=tf.float32)
                    if seq_lengths is not None
                    else tf.ones([batch, max_time]))
        seq_mask = tf.expand_dims(seq_mask, axis=1)  # [batch, 1, max_time]
        sample_level_log_probs = tf.reduce_sum(seq_mask * time_level_log_probs,
                                               axis=2)  # [batch, nsamples]
        return sample_level_log_probs

    def sample(self, inputs, src_enc, input_lengths, src_lengths,
               nsamples=tf.constant(1), random=tf.constant(True)) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        :param inputs: [batch, tgt_max_time, in_dim]
        :param src_enc: [batch, src_max_time, emb_dim]
        :param input_lengths: [batch, ]
        :param src_lengths: [batch, ]
        :param nsamples:
        :param random:
        :return:
        tensor1: samples from the posterior, [batch, nsamples, tgt_max_time, dim]
        tensor2: log-probabilities, [batch, nsamples]
        """
        raise NotImplementedError


class TransformerPosterior(BasePosterior):
    def __init__(self, pre_hidden, pre_drop_rate, pre_activation,
                 pos_drop_rate, nblk, attention_dim, attention_heads,
                 temperature, ffn_hidden, latent_dim, name='TransformerPosterior'):
        super(TransformerPosterior, self).__init__(name=name)
        self.pos_weight = tf.Variable(1.0, trainable=True)
        self.prenet = PreNet(units=pre_hidden, drop_rate=pre_drop_rate,
                             activation=pre_activation, name='decoder_prenet')
        self.pe = PositionalEncoding('EncoderPositionEncoding')
        self.pe_dropout = tf.keras.layers.Dropout(rate=pos_drop_rate)
        self.attentions = []
        for i in range(nblk):
            attention = CrossAttentionBLK(input_dim=pre_hidden,
                                          attention_dim=attention_dim,
                                          attention_heads=attention_heads,
                                          attention_temperature=temperature,
                                          ffn_hidden=ffn_hidden)
            self.attentions.append(attention)
        self.mu_projection = tf.keras.layers.Dense(latent_dim,
                                                   kernel_initializer='zeros',
                                                   name='mu_projection')
        self.logvar_projection = tf.keras.layers.Dense(latent_dim,
                                                       kernel_initializer='zeros',
                                                       name='logvar_projection')

    def call(self, inputs, src_enc, src_lengths=None, target_lengths=None, training=None):
        print('tracing back at posterior call')
        prenet_outs = self.prenet(inputs)
        max_time = tf.shape(prenet_outs)[1]
        dim = tf.shape(prenet_outs)[2]
        pos = self.pe.positional_encoding(max_time, dim)
        pos_embs = prenet_outs + self.pos_weight * pos
        pos_embs = self.pe_dropout(pos_embs, training=training)
        att_outs = pos_embs
        for att in self.attentions:
            att_outs, alignments = att(
                inputs=att_outs, memory=src_enc, query_lengths=target_lengths,
                memory_lengths=src_lengths, training=training)
        mu = self.mu_projection(att_outs)
        logvar = self.logvar_projection(att_outs)
        return mu, logvar, None

    def sample(self, inputs, src_enc, input_lengths, src_lengths,
               nsamples=tf.constant(1), random=tf.constant(True), training=None):
        mu, logvar, _ = self.call(inputs, src_enc, input_lengths, src_lengths,
                                  training=training)
        samples, eps = self.reparameterize(mu, logvar, nsamples, random)
        log_probs = self.log_probability(mu, logvar, eps, input_lengths)
        return samples, log_probs
