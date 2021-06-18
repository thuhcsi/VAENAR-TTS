import tensorflow as tf
from typing import Tuple
from modules.utils import PostNet
from modules.attention import BahdanauAttention, CrossAttentionBLK


class BaseDecoder(tf.keras.layers.Layer):
    """ P(y|x,z): decode target sequence from latent variables conditioned by x
    """

    def __init__(self, name='Decoder', **kwargs):
        super(BaseDecoder, self).__init__(name=name, **kwargs)

    def call(self, inputs, text_embd, z_lengths=None, text_lengths=None,
             targets=None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        :param inputs: latent representations, [batch, max_audio_time, z_hidden]
        :param text_embd: text encodings, [batch, max_text_time, T_emb_hidden]
        :param z_lengths: [batch, ]
        :param text_lengths: [batch, ]
        :param targets: [batch, max_audio_time, out_dim]
        :return: tensor1: reconstructed acoustic features, tensor2: alignments
        """
        raise NotImplementedError

    @staticmethod
    def _compute_l1_loss(reconstructed, targets, lengths=None):
        if lengths is not None:
            max_time = tf.shape(targets)[1]
            seq_mask = tf.sequence_mask(lengths, max_time, dtype=tf.float32)
            l1_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.reduce_mean(
                        tf.abs(reconstructed - targets),
                        axis=-1) * seq_mask,
                    axis=-1) / tf.cast(lengths, tf.float32))
        else:
            l1_loss = tf.losses.MeanAbsoluteError(reconstructed, targets)
        return l1_loss

    @staticmethod
    def _compute_l2_loss(reconstructed, targets, lengths=None):
        if lengths is not None:
            max_time = tf.shape(targets)[1]
            seq_mask = tf.sequence_mask(lengths, max_time, dtype=tf.float32)
            l2_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.reduce_mean(
                        tf.square(reconstructed - targets),
                        axis=-1) * seq_mask,
                    axis=-1) / tf.cast(lengths, tf.float32))
        else:
            l2_loss = tf.losses.MeanSquaredError(reconstructed, targets)
        return l2_loss


class TacotronDecoder(BaseDecoder):
    def __init__(self, in_lstm_hidden, attention_dim, attention_temperature,
                 dec_n_lstm, dec_lstm_hidden, n_conv, conv_filters,
                 conv_kernel, out_dim, drop_rate, name='PostNetDecoder', **kwargs):
        super(TacotronDecoder, self).__init__(name=name, **kwargs)
        self.input_lstm = tf.keras.layers.LSTM(units=in_lstm_hidden,
                                               return_sequences=True)
        self.attention1 = BahdanauAttention(attention_dim=attention_dim,
                                            temperature=attention_temperature,
                                            name='dec_attention_1')
        self.dec_lstm_stack = []
        for i in range(dec_n_lstm):
            lstm = tf.keras.layers.LSTM(units=dec_lstm_hidden,
                                        return_sequences=True,
                                        name='dec_lstm_{}'.format(i))
            self.dec_lstm_stack.append(lstm)
        self.attention2 = BahdanauAttention(attention_dim=attention_dim,
                                            temperature=attention_temperature,
                                            name='dec_attention_2')
        self.pre_projection = tf.keras.layers.Dense(units=out_dim, name='pre_projection')
        self.postnet = PostNet(n_conv=n_conv, conv_filters=conv_filters,
                               conv_kernel=conv_kernel, drop_rate=drop_rate,
                               name='decoder_postnet')
        self.post_projection = tf.keras.layers.Dense(units=out_dim, name='post_projection')

    def call(self, inputs, text_embd, z_lengths=None,
             text_lengths=None, targets=None, training=None):
        """ z -> lstm -> attention -> concat -> 2 * lstm
             -> attention - > concat -> projection -> postnet -> projection
        :param inputs:
        :param text_embd:
        :param z_lengths:
        :param text_lengths:
        :param targets:
        :return:
        """
        initial_lstm_outs = self.input_lstm(inputs)
        contexts1, alignments1 = self.attention1(
            initial_lstm_outs, text_embd, text_lengths)
        dec_lstm_outs = tf.concat([initial_lstm_outs, contexts1], axis=2)
        for lstm in self.dec_lstm_stack:
            dec_lstm_outs = lstm(dec_lstm_outs)
        contexts2, alignments2 = self.attention2(
            dec_lstm_outs, text_embd, text_lengths)
        projection_inputs = tf.concat([dec_lstm_outs, contexts2], axis=2)
        pre_predictions = self.pre_projection(projection_inputs)
        postnet_outs = self.postnet(pre_predictions, training=training)
        residual = self.post_projection(postnet_outs)
        post_predictions = pre_predictions + residual
        if targets is not None:
            pre_loss = self._compute_l2_loss(pre_predictions, targets, lengths=z_lengths)
            post_loss = self._compute_l2_loss(post_predictions, targets, lengths=z_lengths)
        else:
            pre_loss = None
            post_loss = None
        return post_predictions, [pre_loss, post_loss], [alignments1, alignments2]


class LSTMDecoder(BaseDecoder):
    """
    Conv1D -> Attention -> 2 * LSTM -> Projection
    """

    def __init__(self, in_lstm_hidden, attention_dim, attention_temperature,
                 n_dec_lstm, dec_lstm_hidden, out_dim,
                 name='LSTMDecoder', **kwargs):
        super(LSTMDecoder, self).__init__(name=name, **kwargs)
        self.initial_lstm = tf.keras.layers.LSTM(units=in_lstm_hidden,
                                                 return_sequences=True,
                                                 name='initial_lstm')
        self.attention_layer = BahdanauAttention(attention_dim=attention_dim,
                                                 temperature=attention_temperature,
                                                 name='lstm_decoder_attention')
        self.dec_lstm_stack = []
        for i in range(n_dec_lstm):
            lstm = tf.keras.layers.LSTM(units=dec_lstm_hidden,
                                        return_sequences=True,
                                        name='dec_lstm_{}'.format(i))
            self.dec_lstm_stack.append(lstm)
        self.out_projection = tf.keras.layers.Dense(units=out_dim,
                                                    name='output_projection')

    def call(self, inputs, text_embd, z_lengths=None,
             text_lengths=None, targets=None):
        initial_lstm_out = self.initial_lstm(inputs)
        contexts, alignments = self.attention_layer(
            initial_lstm_out, text_embd, text_lengths)
        dec_lstm_out = tf.concat([initial_lstm_out, contexts], axis=2)
        for lstm in self.dec_lstm_stack:
            dec_lstm_out = lstm(dec_lstm_out)
        proj_inputs = tf.concat([dec_lstm_out, contexts], axis=2)
        outputs = self.out_projection(proj_inputs)
        if targets is not None:
            l2_loss = self._compute_l2_loss(outputs, targets, lengths=z_lengths)
        else:
            l2_loss = None
        return outputs, l2_loss, alignments


class TransformerDecoder(BaseDecoder):
    def __init__(self, nblk, attention_dim, attention_heads,
                 temperature, ffn_hidden, post_n_conv, post_conv_filters,
                 post_conv_kernel, post_drop_rate, out_dim, max_reduction_factor,
                 name='TransformerDecoder'):
        super(TransformerDecoder, self).__init__(name=name)
        self.max_reduction_factor = max_reduction_factor
        self.out_dim = out_dim
        self.pre_projection = tf.keras.layers.Dense(units=attention_dim, name='pre_projection')
        self.attentions = []
        for i in range(nblk):
            attention = CrossAttentionBLK(
                input_dim=attention_dim,
                attention_dim=attention_dim,
                attention_heads=attention_heads,
                attention_temperature=temperature,
                ffn_hidden=ffn_hidden, name='decoder-attention-{}'.format(i))
            self.attentions.append(attention)
        self.out_projection = tf.keras.layers.Dense(units=out_dim * self.max_reduction_factor,
                                                    name='linear_outputs')
        self.postnet = PostNet(n_conv=post_n_conv, conv_filters=post_conv_filters,
                               conv_kernel=post_conv_kernel, drop_rate=post_drop_rate,
                               name='postnet')
        self.residual_projection = tf.keras.layers.Dense(units=out_dim, name='residual_outputs')

    def call(self, inputs, text_embd, z_lengths=None, text_lengths=None, reduction_factor=2, training=None):
        print('Tracing back at Self-attention decoder')
        # shape info
        batch_size = tf.shape(inputs)[0]
        max_len = tf.shape(inputs)[1]
        att_outs = self.pre_projection(inputs)
        alignemnts = {}
        for att in self.attentions:
            att_outs, ali = att(
                inputs=att_outs, memory=text_embd, query_lengths=z_lengths,
                memory_lengths=text_lengths, training=training)
            alignemnts[att.name] = ali
        initial_outs = self.out_projection(att_outs)[:, :, : reduction_factor * self.out_dim]
        initial_outs = tf.reshape(initial_outs,
                                  [batch_size, max_len * reduction_factor, self.out_dim])
        residual = self.postnet(initial_outs, training=training)
        residual = self.residual_projection(residual)
        outputs = residual + initial_outs
        return initial_outs, outputs, alignemnts
