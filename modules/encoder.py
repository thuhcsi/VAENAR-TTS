import tensorflow as tf
from typing import Tuple
from .utils import Conv1D, ConvPreNet, PositionalEncoding
from .attention import SelfAttentionBLK


class BaseEncoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embd_dim, name='TextEncoder', **kwargs):
        super(BaseEncoder, self).__init__(name=name, **kwargs)
        self.emb_layer = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                   output_dim=embd_dim,
                                                   name='text_init_encoding')

    def call(self, inputs, input_lengths=None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        :param inputs: text inputs, [batch, max_time]
        :param input_lengths: text inputs' lengths, [batch]
        :return: (tensor1, tensor2)
                tensor1: text encoding, [batch, max_time, hidden_size]
                tensor2: global state, i.e., final_time_state, [batch, hidden_size]
        """
        raise NotImplementedError


class TacotronEncoder(BaseEncoder):
    def __init__(self, vocab_size, embd_dim, n_conv, conv_filter,
                 conv_kernel, conv_activation, lstm_hidden, drop_rate,
                 bn_before_act, name='TacotronEncoder', **kwargs):
        super(TacotronEncoder, self).__init__(vocab_size, embd_dim,
                                              name=name, **kwargs)
        self.conv_stack = []
        for i in range(n_conv):
            conv = Conv1D(filters=conv_filter, kernel_size=conv_kernel,
                          padding='SAME', activation=conv_activation, drop_rate=drop_rate,
                          bn_before_act=bn_before_act, name='conv_{}'.format(i))
            self.conv_stack.append(conv)
        self.blstm_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units=lstm_hidden, return_sequences=True),
            merge_mode='concat', name='blstm_layer')

    def call(self, inputs, input_lengths=None, training=None):
        print('tracing back at text encoding')
        embs = self.emb_layer(inputs)
        conv_out = embs
        for conv in self.conv_stack:
            conv_out = conv(conv_out, training=training)
        blstm_out = self.blstm_layer(conv_out)
        # batch_size = tf.shape(inputs)[0]
        # batch_idx = tf.range(batch_size, dtype=tf.int32, name='batch_idx')
        # time_idx = (input_lengths - 1 if input_lengths is not None
        #             else tf.zeros([batch_size, ], dtype=tf.int32) - 1)
        # indices = tf.stack([batch_idx, time_idx], axis=1)
        # ctx = tf.gather_nd(blstm_out, indices)
        # return blstm_out, ctx
        return blstm_out


class TransformerEncoder(BaseEncoder):
    def __init__(self, vocab_size, embd_dim, pre_nconv, pre_hidden, pre_conv_kernel,
                 prenet_drop_rate, pre_activation, bn_before_act, pos_drop_rate, nblk,
                 attention_dim, attention_heads, attention_temperature, ffn_hidden,
                 name='TextEncoder', **kwargs):
        super(TransformerEncoder, self).__init__(vocab_size, embd_dim, name=name, **kwargs)
        self.pos_weight = tf.Variable(1.0, trainable=True)
        self.prenet = ConvPreNet(nconv=pre_nconv, hidden=pre_hidden,
                                 conv_kernel=pre_conv_kernel, drop_rate=prenet_drop_rate,
                                 activation=pre_activation, bn_before_act=bn_before_act,
                                 name='EncoderPrenet')
        self.pe = PositionalEncoding('EncoderPositionEncoding')
        self.pe_dropout = tf.keras.layers.Dropout(rate=pos_drop_rate)
        self.self_attentions = []
        for i in range(nblk):
            att = SelfAttentionBLK(
                input_dim=pre_hidden, attention_dim=attention_dim,
                attention_heads=attention_heads, attention_temperature=attention_temperature,
                ffn_hidden=ffn_hidden, name='self_attention{}'.format(i))
            self.self_attentions.append(att)

    def call(self, inputs, input_lengths=None, pos_step=1.0, training=None):
        print('tracing back at text encoding')
        embs = self.emb_layer(inputs)
        prenet_outs = self.prenet(embs, training=training)
        max_time = tf.shape(prenet_outs)[1]
        dim = tf.shape(prenet_outs)[2]
        pos = self.pe.positional_encoding(max_time, dim, pos_step)
        pos_embs = prenet_outs + self.pos_weight * pos
        pos_embs = self.pe_dropout(pos_embs, training=training)
        att_outs = pos_embs
        for att in self.self_attentions:
            att_outs, alignments = att(
                inputs=att_outs, memory=att_outs, query_lengths=input_lengths,
                memory_lengths=input_lengths, training=training)
        return att_outs
