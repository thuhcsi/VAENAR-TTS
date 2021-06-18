import tensorflow as tf


class PreNet(tf.keras.layers.Layer):
    def __init__(self, units, drop_rate, activation, name='PreNet', **kwargs):
        super(PreNet, self).__init__(name=name, **kwargs)
        self.dense1 = tf.keras.layers.Dense(
            units=units, activation=activation, name='dense_1')
        self.dense2 = tf.keras.layers.Dense(
            units=units, activation=activation, name='dense_2')
        self.dropout_layer = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, inputs, training=None):
        dense1_out = self.dense1(inputs)
        dense1_out = self.dropout_layer(dense1_out, training=training)
        dense2_out = self.dense2(dense1_out)
        dense2_out = self.dropout_layer(dense2_out, training=training)
        return dense2_out


class ConvPreNet(tf.keras.layers.Layer):
    def __init__(self, nconv, hidden, conv_kernel, drop_rate,
                 activation=tf.nn.relu, bn_before_act=True, name='ConvPrenet', **kwargs):
        super(ConvPreNet, self).__init__(name=name, **kwargs)
        self.conv_stack = []
        for i in range(nconv):
            conv = Conv1D(filters=hidden, kernel_size=conv_kernel, activation=activation,
                          drop_rate=drop_rate, bn_before_act=bn_before_act,
                          name='PreNetConv{}'.format(i))
            self.conv_stack.append(conv)
        self.projection = tf.keras.layers.Dense(units=hidden)

    def call(self, inputs, training=None):
        conv_outs = inputs
        for conv in self.conv_stack:
            conv_outs = conv(conv_outs, training=training)
        projections = self.projection(conv_outs)
        return projections


class FFN(tf.keras.layers.Layer):
    def __init__(self, hidden1, hidden2, name='PositionalFeedForward', **kwargs):
        super(FFN, self).__init__(name=name, **kwargs)
        self.dense1 = tf.keras.layers.Dense(units=hidden1, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units=hidden2, activation=None)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs, training=None):
        dense1_outs = self.dense1(inputs)
        dense2_outs = self.dense2(dense1_outs)
        outs = dense2_outs + inputs
        outs = self.layer_norm(outs, training=training)
        return outs


class Conv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, activation, drop_rate,
                 bn_before_act=False, padding='SAME', strides=1,
                 name='Conv1D_with_dropout_BN', **kwargs):
        super(Conv1D, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.drop_rate = drop_rate
        self.padding = padding
        self.conv1d = tf.keras.layers.Conv1D(filters=filters,
                                             kernel_size=kernel_size,
                                             strides=strides,
                                             padding=padding,
                                             activation=None,
                                             name='conv1d')
        self.activation = activation if activation is not None else tf.identity
        self.bn = tf.keras.layers.BatchNormalization(name='batch_norm')
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate, name='dropout')
        self.bn_before_act = bn_before_act

    def call(self, inputs, training=None):
        conv_outs = self.conv1d(inputs)
        if self.bn_before_act:
            conv_outs = self.bn(conv_outs, training=training)
            conv_outs = self.activation(conv_outs)
        else:
            conv_outs = self.activation(conv_outs)
            conv_outs = self.bn(conv_outs, training=training)
        dropouts = self.dropout(conv_outs, training=training)
        return dropouts

    def get_config(self):
        config = super(Conv1D, self).get_config()
        config.update({'filters': self.filters,
                       'kernel_size': self.kernel_size,
                       'padding': self.padding,
                       'activation': self.activation,
                       'dropout_rate': self.drop_rate,
                       'bn_before_act': self.bn_before_act})
        return config


class PostNet(tf.keras.layers.Layer):
    def __init__(self, n_conv, conv_filters, conv_kernel,
                 drop_rate, name='PostNet', **kwargs):
        super(PostNet, self).__init__(name=name, **kwargs)
        self.conv_stack = []
        self.batch_norm_stack = []
        activations = [tf.math.tanh] * (n_conv - 1) + [tf.identity]
        for i in range(n_conv):
            conv = Conv1D(filters=conv_filters, kernel_size=conv_kernel,
                          padding='same', activation=activations[i],
                          drop_rate=drop_rate, name='conv_{}'.format(i))
            self.conv_stack.append(conv)

    def call(self, inputs, training=None):
        conv_out = inputs
        for conv in self.conv_stack:
            conv_out = conv(conv_out, training)
        return conv_out


class HighwayLayer(tf.keras.layers.Layer):
    def __init__(self, out_dim, name='highwaylayer', **kwargs):
        super(HighwayLayer, self).__init__(name=name, **kwargs)
        self.out_dim = out_dim
        self.relu_layer = tf.keras.layers.Dense(units=self.out_dim,
                                                activation=tf.nn.relu,
                                                name='highway_relu')
        self.sigmoid_layer = tf.keras.layers.Dense(units=self.out_dim,
                                                   activation=tf.nn.sigmoid,
                                                   name='highway_sigmoid')

    def call(self, inputs):
        out = self.relu_layer(inputs) * self.sigmoid_layer(inputs) + inputs * (1.0 - self.sigmoid_layer(inputs))
        return out


class CBHGLayer(tf.keras.layers.Layer):
    def __init__(self, n_convbank, bank_filters, proj_filters, proj_kernel,
                 n_highwaylayer, highway_out_dim, gru_hidden, drop_rate,
                 bn_before_act, name='CBHG_Layer', **kwargs):
        super(CBHGLayer, self).__init__(name=name, **kwargs)
        self.conv_bank = []
        for i in range(n_convbank):
            conv_layer = Conv1D(filters=bank_filters, kernel_size=i + 1,
                                activation=tf.nn.relu, padding='SAME',
                                drop_rate=drop_rate, bn_before_act=bn_before_act,
                                name='conv_layer_{}'.format(i))
            self.conv_bank.append(conv_layer)
        self.maxpooling_layer = tf.keras.layers.MaxPool1D(pool_size=2, strides=1,
                                                          padding='same')
        self.proj_layer1 = Conv1D(filters=proj_filters, kernel_size=proj_kernel,
                                  strides=1, padding='same', activation=tf.nn.relu,
                                  drop_rate=drop_rate, bn_before_act=bn_before_act,
                                  name='projection1')
        self.proj_layer2 = Conv1D(filters=proj_filters, kernel_size=proj_kernel,
                                  strides=1, padding='same', activation=None,
                                  drop_rate=drop_rate, bn_before_act=bn_before_act,
                                  name='projection2')
        self.highway_layers = []
        for i in range(n_highwaylayer):
            hl = HighwayLayer(out_dim=highway_out_dim,
                              name='highway{}'.format(i))
            self.highway_layers.append(hl)
        self.bi_gru_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units=gru_hidden, return_sequences=True),
            merge_mode='concat', name='Bi-GRU')

    def __call__(self, inputs, training=None):
        # 1. convolution bank
        convbank_outs = tf.concat(
            [conv_layer(inputs, training) for conv_layer in self.conv_bank], axis=-1)

        # 2. maxpooling
        maxpool_out = self.maxpooling_layer(convbank_outs)

        # 3. projection layers
        proj1_out = self.proj_layer1(maxpool_out, training)
        proj2_out = self.proj_layer2(proj1_out, training)

        # 4. residual connections
        highway_inputs = proj2_out + inputs

        # 5. highway layers
        for layer in self.highway_layers:
            highway_inputs = layer(highway_inputs)

        # 6. bidirectional GRU
        final_outs = self.bi_gru_layer(highway_inputs)
        return final_outs


class CBHLayer(tf.keras.layers.Layer):
    def __init__(self, n_convbank, bank_filters, proj_filters, proj_kernel,
                 n_highwaylayer, highway_out_dim, drop_rate, bn_before_act,
                 name='CBHG_Layer', **kwargs):
        super(CBHLayer, self).__init__(name=name, **kwargs)
        self.conv_bank = []
        for i in range(n_convbank):
            conv_layer = Conv1D(filters=bank_filters, kernel_size=i + 1,
                                activation=tf.nn.relu, padding='SAME',
                                drop_rate=drop_rate, bn_before_act=bn_before_act,
                                name='conv_layer_{}'.format(i))
            self.conv_bank.append(conv_layer)
        self.maxpooling_layer = tf.keras.layers.MaxPool1D(pool_size=2, strides=1,
                                                          padding='same')
        self.proj_layer1 = Conv1D(filters=proj_filters, kernel_size=proj_kernel,
                                  strides=1, padding='same', activation=tf.nn.relu,
                                  drop_rate=drop_rate, bn_before_act=bn_before_act,
                                  name='projection1')
        self.proj_layer2 = Conv1D(filters=proj_filters, kernel_size=proj_kernel,
                                  strides=1, padding='same', activation=None,
                                  drop_rate=drop_rate, bn_before_act=bn_before_act,
                                  name='projection2')
        self.highway_layers = []
        for i in range(n_highwaylayer):
            hl = HighwayLayer(out_dim=highway_out_dim,
                              name='highway{}'.format(i))
            self.highway_layers.append(hl)

    def __call__(self, inputs, training=None):
        # 1. convolution bank
        convbank_outs = tf.concat(
            [conv_layer(inputs, training) for conv_layer in self.conv_bank], axis=-1)

        # 2. maxpooling
        maxpool_out = self.maxpooling_layer(convbank_outs)

        # 3. projection layers
        proj1_out = self.proj_layer1(maxpool_out, training)
        proj2_out = self.proj_layer2(proj1_out, training)

        # 4. residual connections
        highway_inputs = proj2_out + inputs

        # 5. highway layers
        for layer in self.highway_layers:
            highway_inputs = layer(highway_inputs)
        final_outs = highway_inputs
        return final_outs


class DilatedConv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding,
                 dilations, name='DilatedConv1D', **kwargs):
        super(DilatedConv1D, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilations = dilations

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.kernel = self.add_weight(
            name='kernel', shape=[self.kernel_size, int(in_channels), self.filters],
            dtype=tf.float32, initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True)
        self.bias = self.add_weight(
            name='bias', shape=[self.filters, ], dtype=tf.float32,
            initializer='zeros', trainable=True)

    def call(self, inputs):
        conved = tf.nn.conv1d(inputs, self.kernel, stride=1, padding=self.padding,
                              dilations=self.dilations)
        conved = conved + self.bias
        return conved

    def get_config(self):
        config = super(DilatedConv1D, self).get_config()
        config.update({'filters': self.filters,
                       'kernel_size': self.kernel_size,
                       'dilations': self.dilations,
                       'padding': self.padding})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class DCNResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernels, dilation_rate, drop_rate,
                 name='DCNResidualBlock', **kwargs):
        """
        :param filters: should be the same with the last dimension of inputs
        :param kernels:
        :param dilation_rate:
        :param drop_rate:
        :param name:
        :param kwargs:
        """
        super(DCNResidualBlock, self).__init__(name=name, **kwargs)
        self.conv1 = DilatedConv1D(filters=filters, kernel_size=kernels, padding='SAME',
                                   dilations=dilation_rate, name='dilated_conv1')
        self.conv2 = DilatedConv1D(filters=filters, kernel_size=kernels, padding='SAME',
                                   dilations=dilation_rate, name='dilated_conv2')
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)
        self.batch_norm = tf.keras.layers.BatchNormalization(name='bactch_norm')

    def call(self, inputs, training=None):
        """
        :param inputs: [batch, time, filters]
        :param training
        :return:
        """
        conv1_out = self.conv1(inputs)
        conv1_out = self.dropout(tf.nn.relu(conv1_out), training=training)
        conv2_out = self.conv2(conv1_out)
        conv2_out = self.dropout(tf.nn.relu(conv2_out), training=training)
        return self.batch_norm(inputs + conv2_out)


class DCNModule(tf.keras.layers.Layer):
    def __init__(self, n_block, filters, kernels, drop_rate, name='DCNModule', **kwargs):
        super(DCNModule, self).__init__(name=name, **kwargs)
        self.dcn_stack = []
        for i in range(n_block):
            dcn_blk = DCNResidualBlock(filters=filters,
                                       kernels=kernels,
                                       dilation_rate=2 ** i,
                                       drop_rate=drop_rate,
                                       name='DCN_residual_blk{}'.format(i), )
            self.dcn_stack.append(dcn_blk)

    def call(self, inputs, training=None):
        outputs = inputs
        for blk in self.dcn_stack:
            outputs = blk(outputs, training=training)
        return outputs


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, name='PositionalEncoding'):
        super(PositionalEncoding, self).__init__(name=name)

    @staticmethod
    def positional_encoding(len, dim, step=1.):
        """
        :param len: int scalar
        :param dim: int scalar
        :param step:
        :return: position embedding
        """
        pos_mat = tf.tile(
            tf.expand_dims(
                tf.range(0, tf.cast(len, dtype=tf.float32), dtype=tf.float32) * step,
                axis=-1),
            [1, dim])
        dim_mat = tf.tile(
            tf.expand_dims(
                tf.range(0, tf.cast(dim, dtype=tf.float32), dtype=tf.float32),
                axis=0),
            [len, 1])
        dim_mat_int = tf.cast(dim_mat, dtype=tf.int32)
        pos_encoding = tf.where(  # [time, dims]
            tf.math.equal(tf.math.mod(dim_mat_int, 2), 0),
            x=tf.math.sin(pos_mat / tf.pow(10000., dim_mat / tf.cast(dim, tf.float32))),
            y=tf.math.cos(pos_mat / tf.pow(10000., (dim_mat - 1) / tf.cast(dim, tf.float32))))
        return pos_encoding
