import tensorflow as tf
from modules.encoder import TransformerEncoder
from modules.posterior import TransformerPosterior
from modules.decoder import TransformerDecoder
from modules.prior import TransformerPrior
from modules.length_predictor import DenseLengthPredictor


class VAENAR(tf.keras.Model):
    def __init__(self, hps, name='VAENAR', **kwargs):
        super(VAENAR, self).__init__(name=name, **kwargs)
        self.hps = hps
        self.n_sample = hps.Train.num_samples
        self.mel_text_len_ratio = hps.Common.mel_text_len_ratio
        self.max_reduction_factor = hps.Common.max_reduction_factor
        self.text_encoder = TransformerEncoder(
            vocab_size=hps.Encoder.Transformer.vocab_size,
            embd_dim=hps.Encoder.Transformer.embd_dim,
            pre_nconv=hps.Encoder.Transformer.n_conv,
            pre_hidden=hps.Encoder.Transformer.pre_hidden,
            pre_conv_kernel=hps.Encoder.Transformer.conv_kernel,
            pre_activation=hps.Encoder.Transformer.pre_activation,
            prenet_drop_rate=hps.Encoder.Transformer.pre_drop_rate,
            bn_before_act=hps.Encoder.Transformer.bn_before_act,
            pos_drop_rate=hps.Encoder.Transformer.pos_drop_rate,
            nblk=hps.Encoder.Transformer.n_blk,
            attention_dim=hps.Encoder.Transformer.attention_dim,
            attention_heads=hps.Encoder.Transformer.attention_heads,
            attention_temperature=hps.Encoder.Transformer.attention_temperature,
            ffn_hidden=hps.Encoder.Transformer.ffn_hidden, )
        self.decoder = TransformerDecoder(
            nblk=hps.Decoder.Transformer.nblk,
            attention_dim=hps.Decoder.Transformer.attention_dim,
            attention_heads=hps.Decoder.Transformer.attention_heads,
            temperature=hps.Decoder.Transformer.attention_temperature,
            ffn_hidden=hps.Decoder.Transformer.ffn_hidden,
            post_n_conv=hps.Decoder.Transformer.post_n_conv,
            post_conv_filters=hps.Decoder.Transformer.post_conv_filters,
            post_conv_kernel=hps.Decoder.Transformer.post_conv_kernel,
            post_drop_rate=hps.Decoder.Transformer.post_drop_rate,
            out_dim=hps.Common.output_dim,
            max_reduction_factor=hps.Common.max_reduction_factor,
            name='transformer_decoder')
        self.length_predictor = DenseLengthPredictor(
            activation=hps.LengthPredictor.Dense.activation)
        self.posterior = TransformerPosterior(
            pre_hidden=hps.Posterior.Transformer.pre_hidden,
            pos_drop_rate=hps.Posterior.Transformer.pos_drop_rate,
            pre_drop_rate=hps.Posterior.Transformer.pre_drop_rate,
            pre_activation=hps.Posterior.Transformer.pre_activation,
            nblk=hps.Posterior.Transformer.nblk,
            attention_dim=hps.Posterior.Transformer.attention_dim,
            attention_heads=hps.Posterior.Transformer.attention_heads,
            temperature=hps.Posterior.Transformer.temperature,
            ffn_hidden=hps.Posterior.Transformer.ffn_hidden,
            latent_dim=hps.Common.latent_dim)
        self.prior = TransformerPrior(
            n_blk=hps.Prior.Transformer.n_blk,
            channels=hps.Common.latent_dim,
            n_transformer_blk=hps.Prior.Transformer.n_transformer_blk,
            attention_dim=hps.Prior.Transformer.attention_dim,
            attention_heads=hps.Prior.Transformer.attention_heads,
            temperature=hps.Prior.Transformer.temperature,
            ffn_hidden=hps.Prior.Transformer.ffn_hidden,
            inverse=hps.Prior.Transformer.inverse)

    def _compute_l2_loss(self, reconstructed, targets, lengths=None, reduce=None):
        max_time = tf.shape(reconstructed)[1]
        dim = tf.shape(reconstructed)[2]
        r = tf.reshape(reconstructed, [-1, self.n_sample, max_time, dim])
        t = tf.reshape(targets, [-1, self.n_sample, max_time, dim])
        if lengths is not None:
            seq_mask = tf.sequence_mask(lengths, max_time, dtype=tf.float32)
            seq_mask = tf.reshape(seq_mask, [-1, self.n_sample, max_time])
            reshaped_lens = tf.reshape(lengths, [-1, self.n_sample])
            l2_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.reduce_mean(tf.square(r - t), axis=-1) * seq_mask,
                    axis=-1) / tf.cast(reshaped_lens, tf.float32),
                axis=-1)
        else:
            l2_loss = tf.math.reduce_mean(tf.square(r - t), axis=[1, 2, 3])
        if reduce:
            return tf.math.reduce_mean(l2_loss)
        else:
            return l2_loss

    @staticmethod
    def _kl_divergence(p, q, reduce=None):
        kl = tf.math.reduce_mean((p - q), axis=1)
        if reduce:
            return tf.math.reduce_mean(kl)
        else:
            return kl

    @staticmethod
    def _length_l2_loss(predicted_lengths, target_lengths, reduce=None):
        log_tgt_lengths = tf.math.log(tf.cast(target_lengths, tf.float32))
        log_pre_lengths = tf.math.log(predicted_lengths)
        if reduce:
            return tf.reduce_mean(tf.square(log_pre_lengths - log_tgt_lengths))
        else:
            return tf.square(log_pre_lengths - log_tgt_lengths)

    def call(self, inputs, mel_targets, mel_lengths, text_lengths=None,
             reduction_factor=2, training=None, reduce_loss=None):
        """
        :param inputs: text inputs, [batch, text_max_time, vocab_size]
        :param mel_lengths: [batch, ]
        :param text_lengths: [batch, ]
        :param training: bool
        :param mel_targets: [batch, mel_max_time, mel_dim]
        :param reduce_loss: bool
        :return: predicted mel: [batch, mel_max_time, mel_dim]
                 loss: float32
        """
        print('tracing back at FlowTacotron.call')
        # shape info
        batch_size = tf.shape(mel_targets)[0]
        mel_max_len = tf.shape(mel_targets)[1]
        text_max_len = tf.shape(inputs)[1]
        # reduce the mels
        reduced_mels = mel_targets[:, ::reduction_factor, :]
        reduced_mels.set_shape([None, None, self.hps.Audio.num_mels])
        reduced_mel_lens = (mel_lengths + reduction_factor - 1) // reduction_factor
        reduced_mel_max_len = tf.shape(reduced_mels)[1]

        # text encoding
        text_pos_step = self.mel_text_len_ratio / tf.cast(reduction_factor, tf.float32)
        text_embd = self.text_encoder(
            inputs, text_lengths, pos_step=text_pos_step, training=training)
        predicted_lengths = self.length_predictor(
            tf.stop_gradient(text_embd), text_lengths, training=training)
        length_loss = self._length_l2_loss(
            predicted_lengths, mel_lengths, reduce=reduce_loss)
        logvar, mu, post_alignments = self.posterior(reduced_mels, text_embd,
                                                     src_lengths=text_lengths,
                                                     target_lengths=reduced_mel_lens,
                                                     training=training)
        # samples, eps: [batch, n_sample, mel_max_time, dim]
        samples, eps = self.posterior.reparameterize(mu, logvar, self.n_sample)
        # [batch, n_sample]
        posterior_logprobs = self.posterior.log_probability(
            mu, logvar, eps=eps, seq_lengths=reduced_mel_lens)
        # [batch*n_sample, mel_max_len, dim]
        batched_samples = tf.reshape(
            samples, [batch_size * self.n_sample, reduced_mel_max_len, -1])
        batched_samples.set_shape([None, None, self.hps.Common.latent_dim])
        # [batch*n_sample, text_max_len, dim]
        batched_text_embd = tf.reshape(
            tf.tile(
                tf.expand_dims(text_embd, axis=1),
                tf.constant([1, self.n_sample, 1, 1])),
            [batch_size * self.n_sample, text_max_len, -1])
        batched_text_embd.set_shape(
            [None, None, self.hps.Encoder.Transformer.embd_dim])
        batched_mel_targets = tf.reshape(
            tf.tile(
                tf.expand_dims(mel_targets, axis=1),
                tf.constant([1, self.n_sample, 1, 1])),
            [batch_size * self.n_sample, mel_max_len, -1])
        batched_mel_targets.set_shape(
            [None, None, self.hps.Audio.num_mels])
        # [batch*n_sample, ]
        batched_mel_lengths = tf.reshape(
            tf.tile(
                tf.expand_dims(mel_lengths, axis=1),
                tf.constant([1, self.n_sample])), [-1])
        # [batch*n_sample, ]
        batched_r_mel_lengths = tf.reshape(
            tf.tile(
                tf.expand_dims(reduced_mel_lens, axis=1),
                tf.constant([1, self.n_sample])), [-1])
        # [batch*n_sample, ]
        batched_text_lengths = tf.reshape(
            tf.tile(
                tf.expand_dims(text_lengths, axis=1),
                tf.constant([1, self.n_sample])), [-1])
        decoded_initial, decoded_outs, dec_alignments = self.decoder(
            batched_samples, batched_text_embd, batched_r_mel_lengths,
            batched_text_lengths, training=training, reduction_factor=reduction_factor)
        decoded_initial = decoded_initial[:, :mel_max_len, :]
        decoded_outs = decoded_outs[:, :mel_max_len, :]
        initial_l2_loss = self._compute_l2_loss(decoded_initial, batched_mel_targets,
                                                batched_mel_lengths, reduce_loss)
        l2_loss = self._compute_l2_loss(decoded_outs, batched_mel_targets,
                                        batched_mel_lengths, reduce_loss)
        l2_loss += initial_l2_loss
        # [batch*n_sample, ]
        prior_logprobs = self.prior.log_probability(z=batched_samples,
                                                    condition_inputs=batched_text_embd,
                                                    z_lengths=batched_r_mel_lengths,
                                                    condition_lengths=batched_text_lengths,
                                                    training=training)
        prior_logprobs = tf.reshape(prior_logprobs, [batch_size, self.n_sample])
        kl_divergence = self._kl_divergence(posterior_logprobs, prior_logprobs, reduce_loss)
        return decoded_outs, l2_loss, kl_divergence, length_loss, dec_alignments

    def inference(self, inputs, mel_lengths, text_lengths=None, reduction_factor=2):
        reduced_mel_lens = (mel_lengths + reduction_factor - 1) // reduction_factor
        text_pos_step = self.mel_text_len_ratio / tf.cast(reduction_factor, tf.float32)
        text_embd = self.text_encoder(inputs, text_lengths, pos_step=text_pos_step, training=False)
        prior_latents, prior_logprobs = self.prior.sample(reduced_mel_lens,
                                                          text_embd,
                                                          text_lengths,
                                                          training=False)
        _, predicted_mel, dec_alignments = self.decoder(
            inputs=prior_latents, text_embd=text_embd, z_lengths=reduced_mel_lens,
            text_lengths=text_lengths, training=False, reduction_factor=reduction_factor)
        return predicted_mel, dec_alignments

    def init(self, text_inputs, mel_lengths, text_lengths=None):
        reduced_mel_lens = (mel_lengths + self.max_reduction_factor - 1) // self.max_reduction_factor
        text_pos_step = self.mel_text_len_ratio / tf.cast(self.max_reduction_factor, tf.float32)
        text_embd = self.text_encoder(text_inputs, text_lengths, pos_step=text_pos_step, training=True)
        prior_latents, prior_logprobs = self.prior.init(conditions=text_embd,
                                                        targets_lengths=reduced_mel_lens,
                                                        condition_lengths=text_lengths,
                                                        training=True)
        _, predicted_mel, _ = self.decoder(inputs=prior_latents,
                                           text_embd=text_embd,
                                           z_lengths=reduced_mel_lens,
                                           text_lengths=text_lengths,
                                           reduction_factor=self.max_reduction_factor,
                                           training=True)
        return predicted_mel
