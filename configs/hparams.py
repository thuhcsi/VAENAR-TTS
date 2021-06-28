import tensorflow as tf


class Hparams:
    class Train:
        random_seed = 12
        epochs = 2000
        warm_epochs = 0
        train_batch_size = 32
        test_batch_size = 8
        test_interval = 50
        shuffle_buffer = 128
        shuffle = True
        num_samples = 1
        length_weight = 1.
        kl_weight = 1.
        kl_weight_init = 1e-5
        kl_weight_increase_epoch = 1
        kl_weight_end = 1e-5
        learning_rate = 1.25e-4
        reduction_factors = [5, 4, 3, 2]
        reduce_interval = [0, 200, 400, 600]

    class Dataset:
        buffer_size = 65536
        num_parallel_reads = 64
        pad_factor = 0  # factor ** (num_blk - 1)
        dev_set_rate = 0.01
        test_set_rate = 0.01

    class Texts:
        pad = '_'
        bos = '^'
        eos = '~'
        characters = '_^~abcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? []'

    class Audio:
        num_mels = 80
        num_freq = 1025
        min_mel_freq = 30.
        max_mel_freq = 7600.
        sample_rate = 22050
        frame_length_sample = 1024
        frame_shift_sample = 256
        n_mfcc = 13
        preemphasize = 0.97
        min_level_db = -100.0
        ref_level_db = 20.0
        max_abs_value = 1
        symmetric_specs = False
        griffin_lim_iters = 60
        power = 1.5
        center = True

    class Common:
        latent_dim = 128
        output_dim = 80
        final_reduction_factor = 2
        max_reduction_factor = 5
        mel_text_len_ratio = 5.59

    class Encoder:
        class Tacotron:
            vocab_size = 43
            embd_dim = 512
            n_conv = 3
            conv_filter = 512
            conv_kernel = 5
            conv_activation = tf.nn.relu
            lstm_hidden = 256
            drop_rate = 0.0
            bn_before_act = False

        class Transformer:
            vocab_size = 43
            embd_dim = 512
            n_conv = 3
            pre_hidden = 512
            conv_kernel = 5
            pre_activation = tf.nn.relu
            pre_drop_rate = 0.1
            pos_drop_rate = 0.1
            bn_before_act = False
            n_blk = 4
            attention_dim = 256
            attention_heads = 4
            attention_temperature = 1.0
            ffn_hidden = 1024

    class Decoder:
        class CBHG:
            lstm_hidden = 128
            attention_dim = 128
            attention_window = 2
            attention_heads = 4
            attention_temperature = 1.0
            n_conv_bank = 8
            bank_filters = 128
            proj_filters = 256
            proj_kernel = 3
            n_highwaylayer = 4
            highway_out_dim = 256
            gru_hidden = 256
            drop_rate = 0.1
            bn_before_act = False

        class Transformer:
            pre_hidden = 128
            pre_drop_rate = 0.5
            pre_activation = tf.nn.relu
            nblk = 2
            attention_dim = 256
            attention_heads = 4
            ffn_hidden = 1024
            attention_temperature = 1.
            post_n_conv = 5
            post_conv_filters = 256
            post_conv_kernel = 5
            post_drop_rate = 0.2

        class ARTransformer:
            max_state_size = 812
            attention_loc_filters = 32
            attention_loc_kernel = 31
            cumulative_weights = True
            attention_window = 2
            nblk = 2
            attention_dim = 256
            attention_heads = 4
            ffn_hidden = 1024
            attention_temperature = 1.
            post_n_conv = 5
            post_conv_filters = 256
            post_conv_kernel = 5
            post_drop_rate = 0.2

    class Posterior:
        class Transformer:
            pre_hidden = 256
            pos_drop_rate = 0.2
            pre_drop_rate = 0.5
            pre_activation = tf.nn.relu
            bn_before_act = False
            nblk = 2
            attention_dim = 256
            attention_heads = 4
            temperature = 1.0
            ffn_hidden = 1024

        class CBHG:
            pre_units = 256
            pre_drop_rate = 0.5
            pre_activation = 'relu'
            lstm_hidden = 128
            attention_dim = 128
            attention_window = 2
            attention_heads = 4
            attention_temperature = 1.0
            n_conv_bank = 8
            bank_filters = 128
            proj_filters = 256
            proj_kernel = 3
            n_highwaylayer = 4
            highway_out_dim = 256
            gru_hidden = 256
            drop_rate = 0.1
            bn_before_act = False

    class Prior:
        class Transformer:
            n_blk = 6
            n_transformer_blk = 2
            attention_dim = 256
            attention_heads = 4
            temperature = 1.0
            ffn_hidden = 1024
            inverse = False

        class CBHG:
            prior_n_blk = 4
            lstm_hidden = 128
            n_conv_bank = 8
            bank_filters = 128
            proj_filters = 256
            proj_kernel = 3
            n_highwaylayer = 4
            highway_out_dim = 256
            gru_hidden = 256
            attention_dim = 128
            drop_rate = 0.1
            inverse = False
            bn_before_act = False

        class MultiscaleCBHG:
            steps = [4, 2, 2]
            factor = 2
            lstm_hidden = 128
            n_conv_bank = 4
            bank_filters = 128
            proj_filters = 256
            proj_kernel = 3
            n_highwaylayer = 2
            highway_out_dim = 256
            gru_hidden = 256
            attention_dim = 128
            drop_rate = 0.1
            inverse = False
            bn_before_act = False

    class Attention:
        attention_dim = 128
        max_state_size = 812
        attention_loc_filters = 32
        attention_loc_kernel = 31
        cumulative_weights = True
        attention_window = 2
        attention_heads = 4
        attention_temperature = 1.0

    class LengthPredictor:
        class Conv:
            n_conv = 2
            conv_filter = 256
            conv_kernel = 5
            drop_rate = 0.4
            activation = tf.nn.relu
            bn_before_act = False

        class Dense:
            activation = tf.identity


class LJHPS:
    class Train:
        random_seed = 123456
        epochs = 2000
        warm_epochs = 0
        train_batch_size = 32
        test_batch_size = 8
        test_interval = 50
        shuffle_buffer = 128
        shuffle = True
        num_samples = 1
        length_weight = 1.
        kl_weight = 1.
        kl_weight_init = 1e-5
        kl_weight_increase_epoch = 1
        kl_weight_end = 1e-5
        learning_rate = 1.25e-4
        reduction_factors = [5, 4, 3, 2]
        reduce_interval = [0, 200, 400, 600]

    class Dataset:
        buffer_size = 65536
        num_parallel_reads = 64
        pad_factor = 0  # factor ** (num_blk - 1)
        dev_set_rate = 0.01
        test_set_rate = 0.01

    class Texts:
        pad = '_'
        bos = '^'
        eos = '~'
        characters = '_^~abcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? []'

    class Audio:
        num_mels = 80
        num_freq = 1025
        min_mel_freq = 0.
        max_mel_freq = 8000.
        sample_rate = 22050
        frame_length_sample = 1024
        frame_shift_sample = 256
        n_mfcc = 13
        preemphasize = 0.97
        min_level_db = -100.0
        ref_level_db = 20.0
        max_abs_value = 1
        symmetric_specs = False
        griffin_lim_iters = 60
        power = 1.5
        center = True

    class Common:
        latent_dim = 128
        output_dim = 80
        final_reduction_factor = 2
        max_reduction_factor = 5
        mel_text_len_ratio = 5.59

    class Encoder:
        class Transformer:
            vocab_size = 43
            embd_dim = 512
            n_conv = 3
            pre_hidden = 512
            conv_kernel = 5
            pre_activation = tf.nn.relu
            pre_drop_rate = 0.1
            pos_drop_rate = 0.1
            bn_before_act = False
            n_blk = 4
            attention_dim = 256
            attention_heads = 4
            attention_temperature = 1.0
            ffn_hidden = 1024

    class Decoder:
        class Transformer:
            pre_hidden = 128
            pre_drop_rate = 0.5
            pre_activation = tf.nn.relu
            nblk = 2
            attention_dim = 256
            attention_heads = 4
            ffn_hidden = 1024
            attention_temperature = 1.
            post_n_conv = 5
            post_conv_filters = 256
            post_conv_kernel = 5
            post_drop_rate = 0.2

    class Posterior:
        class Transformer:
            pre_hidden = 256
            pos_drop_rate = 0.2
            pre_drop_rate = 0.5
            pre_activation = tf.nn.relu
            bn_before_act = False
            nblk = 2
            attention_dim = 256
            attention_heads = 4
            temperature = 1.0
            ffn_hidden = 1024

    class Prior:
        class Transformer:
            n_blk = 6
            n_transformer_blk = 2
            attention_dim = 256
            attention_heads = 4
            temperature = 1.0
            ffn_hidden = 1024
            inverse = False

    class LengthPredictor:
        class Dense:
            activation = tf.identity


class DataBakerHPS(Hparams):
    class Train:
        random_seed = 12
        epochs = 2000
        warm_epochs = 0
        train_batch_size = 32
        test_batch_size = 8
        test_interval = 50
        shuffle_buffer = 128
        shuffle = True
        num_samples = 1
        length_weight = 1.
        kl_weight = 1.
        kl_weight_init = 1e-5
        kl_weight_increase_epoch = 1
        kl_weight_end = 1e-5
        learning_rate = 1.25e-4
        reduction_factors = [5, 4, 3, 2]
        reduce_interval = [0, 200, 400, 600]

    class Dataset:
        buffer_size = 65536
        num_parallel_reads = 64
        pad_factor = 0  # factor ** (num_blk - 1)
        dev_set_rate = 0.01
        test_set_rate = 0.01

    class Texts:
        pad = '_'
        bos = '^'
        eos = '~'
        characters = '_^~abcdefghijklmnopqrstuvwxyz12345,./- '

    class Audio:
        num_mels = 80
        num_freq = 1025
        min_mel_freq = 0.
        max_mel_freq = 8000.
        sample_rate = 16000
        frame_length_sample = 800
        frame_shift_sample = 200
        n_mfcc = 13
        preemphasize = 0.97
        min_level_db = -115.
        ref_level_db = 20.
        max_abs_value = 1
        symmetric_specs = False
        griffin_lim_iters = 60
        power = 1.5
        center = True

    class Common:
        latent_dim = 128
        output_dim = 80
        final_reduction_factor = 2
        max_reduction_factor = 5
        mel_text_len_ratio = 4.21

    class Encoder:
        class Transformer:
            vocab_size = 39
            embd_dim = 512
            n_conv = 3
            pre_hidden = 512
            conv_kernel = 5
            pre_activation = tf.nn.relu
            pre_drop_rate = 0.1
            pos_drop_rate = 0.1
            bn_before_act = False
            n_blk = 4
            attention_dim = 256
            attention_heads = 4
            attention_temperature = 1.0
            ffn_hidden = 1024

    class Decoder:
        class Transformer:
            pre_hidden = 128
            pre_drop_rate = 0.5
            pre_activation = tf.nn.relu
            nblk = 2
            attention_dim = 256
            attention_heads = 4
            ffn_hidden = 1024
            attention_temperature = 1.
            post_n_conv = 5
            post_conv_filters = 256
            post_conv_kernel = 5
            post_drop_rate = 0.2

    class Posterior:
        class Transformer:
            pre_hidden = 256
            pos_drop_rate = 0.2
            pre_drop_rate = 0.5
            pre_activation = tf.nn.relu
            bn_before_act = False
            nblk = 2
            attention_dim = 256
            attention_heads = 4
            temperature = 1.0
            ffn_hidden = 1024

    class Prior:
        class Transformer:
            n_blk = 6
            n_transformer_blk = 2
            attention_dim = 256
            attention_heads = 4
            temperature = 1.0
            ffn_hidden = 1024
            inverse = False

    class LengthPredictor:
        class Conv:
            n_conv = 2
            conv_filter = 256
            conv_kernel = 5
            drop_rate = 0.4
            activation = tf.nn.relu
            bn_before_act = False

        class Dense:
            activation = tf.identity
