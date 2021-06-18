import tensorflow as tf
from typing import Tuple

from modules.utils import PositionalEncoding
from modules.attention import CrossAttentionBLK


class BaseTransform(tf.keras.layers.Layer):
    def __init__(self, out_dim, name='BaseTransform', **kwargs):
        super(BaseTransform, self).__init__(name=name, **kwargs)
        self.out_dim = out_dim
        self.log_scale_proj = tf.keras.layers.Dense(units=self.out_dim,
                                                    kernel_initializer='zeros',
                                                    name='log_scale_projection')
        self.shift_proj = tf.keras.layers.Dense(units=self.out_dim,
                                                kernel_initializer='zeros',
                                                name='shift_projection')

    def call(self, inputs, condition_inputs, condition_lengths=None
             ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        :param inputs: xa inputs
        :param condition_inputs:
        :param condition_lengths:
        :return: tensor1: log_scale, tensor2: bias
        """
        raise NotImplementedError


class TransformerTransform(BaseTransform):
    def __init__(self, nblk, attention_dim, attention_heads, temperature,
                 ffn_hidden, out_dim, name='BaseTransform', **kwargs):
        super(TransformerTransform, self).__init__(out_dim=out_dim, name=name, **kwargs)
        self.pos_emb_layer = PositionalEncoding()
        self.pos_weight = tf.Variable(1.0, trainable=True)
        self.pre_projection = tf.keras.layers.Dense(units=attention_dim, name='pre_projection')
        self.attentions = []
        for i in range(nblk):
            attention = CrossAttentionBLK(input_dim=attention_dim,
                                          attention_dim=attention_dim,
                                          attention_heads=attention_heads,
                                          attention_temperature=temperature,
                                          ffn_hidden=ffn_hidden)
            self.attentions.append(attention)

    def call(self, inputs, condition_inputs, condition_lengths=None,
             target_lengths=None, training=None):
        att_outs = self.pre_projection(inputs)
        max_time = tf.shape(att_outs)[1]
        dim = tf.shape(att_outs)[2]
        pos_embd = self.pos_emb_layer.positional_encoding(max_time, dim)
        att_outs += self.pos_weight * pos_embd
        for att in self.attentions:
            att_outs, _ = att(inputs=att_outs, memory=condition_inputs,
                              memory_lengths=condition_lengths,
                              query_lengths=target_lengths)
        log_scale = self.log_scale_proj(att_outs)
        shift = self.shift_proj(att_outs)
        return log_scale, shift
