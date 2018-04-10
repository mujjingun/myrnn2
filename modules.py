import tensorflow as tf

def prenet(inputs, is_training, layer_sizes, drop_prob, scope=None):
    x = inputs
    drop_rate = drop_prob if is_training else 0.0
    with tf.variable_scope(scope or 'prenet'):
        for i, size in enumerate(layer_sizes):
            dense = tf.layers.dense(x, units=size, activation=tf.nn.relu, name='dense_%d' % (i+1))
            x = tf.layers.dropout(dense, rate=drop_rate, name='dropout_%d' % (i+1))
    return x

def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
    with tf.variable_scope(scope):
        conv1d_output = tf.layers.conv1d(
            inputs,
            filters=channels,
            kernel_size=kernel_size,
            activation=activation,
            padding='same')
        return tf.layers.batch_normalization(conv1d_output, training=is_training)
    
def cbhg(inputs, input_lengths, is_training, 
        bank_size, bank_channel_size,
        maxpool_width, highway_depth, rnn_size,
        proj_sizes, proj_width, scope):

    batch_size = tf.shape(inputs)[0]
    with tf.variable_scope(scope):
        with tf.variable_scope('conv_bank'):
            # Convolution bank: concatenate on the last axis
            # to stack channels from all convolutions
            conv_fn = lambda k: \
                    conv1d(inputs, k, bank_channel_size, 
                            tf.nn.relu, is_training, 'conv1d_%d' % k)

            conv_outputs = tf.concat(
                [conv_fn(k) for k in range(1, bank_size + 1)], axis=-1,
            )

        # Maxpooling:
        maxpool_output = tf.layers.max_pooling1d(
            conv_outputs,
            pool_size=maxpool_width,
            strides=1,
            padding='same')

        # Two projection layers:
        proj_out = maxpool_output
        for idx, proj_size in enumerate(proj_sizes):
            activation_fn = None if idx == len(proj_sizes) - 1 else tf.nn.relu
            proj_out = conv1d(
                    proj_out, proj_width, proj_size, activation_fn,
                    is_training, 'proj_{}'.format(idx + 1))

        # Residual connection:
        highway_input = proj_out + inputs

        # Handle dimensionality mismatch:
        if highway_input.shape[2] != rnn_size:
            highway_input = tf.layers.dense(highway_input, rnn_size)

        # 4-layer HighwayNet:
        for idx in range(highway_depth):
            highway_input = highwaynet(highway_input, 'highway_%d' % (idx+1))

        rnn_input = highway_input

        cell_fw, cell_bw = tf.contrib.rnn.GRUCell(rnn_size), tf.contrib.rnn.GRUCell(rnn_size)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw,
                rnn_input,
                sequence_length=input_lengths,
                swap_memory=True,
                dtype=tf.float32)
        return tf.concat(outputs, axis=2)    # Concat forward and backward

def highwaynet(inputs, scope):
    highway_dim = int(inputs.get_shape()[-1])

    with tf.variable_scope(scope):
        H = tf.layers.dense(
            inputs,
            units=highway_dim,
            activation=tf.nn.relu,
            name='H')
        T = tf.layers.dense(
            inputs,
            units=highway_dim,
            activation=tf.nn.sigmoid,
            name='T',
            bias_initializer=tf.constant_initializer(-1.0))
        return H * T + inputs * (1.0 - T)

class DecoderPrenetWrapper(tf.contrib.rnn.RNNCell):
    '''Runs RNN inputs through a prenet before sending them to the cell.'''
    def __init__(
            self, cell,
            is_training, prenet_sizes, dropout_prob):
        super(DecoderPrenetWrapper, self).__init__()
        
        self._is_training = is_training
        self._cell = cell
        self.prenet_sizes = prenet_sizes
        self.dropout_prob = dropout_prob

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def call(self, inputs, state):
        prenet_out = prenet(
                inputs, self._is_training,
                self.prenet_sizes, self.dropout_prob, scope='decoder_prenet')

        return self._cell(prenet_out, state)

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)


class ConcatOutputAndAttentionWrapper(tf.contrib.rnn.RNNCell):
    '''Concatenates RNN cell output with the attention context vector.

    This is expected to wrap a cell wrapped with an AttentionWrapper constructed with
    attention_layer_size=None and output_attention=False. Such a cell's state will include an
    "attention" field that is the context vector.
    '''
    def __init__(self, cell):
        super(ConcatOutputAndAttentionWrapper, self).__init__()
        self._cell = cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size + self._cell.state_size.attention

    def call(self, inputs, state):
        output, res_state = self._cell(inputs, state)
        return tf.concat([output, res_state.attention], axis=-1), res_state

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)
    

# Adapted from tf.contrib.seq2seq.GreedyEmbeddingHelper
class TacoTestHelper(tf.contrib.seq2seq.Helper):
    def __init__(self, batch_size, output_dim, r):
        with tf.name_scope('TacoTestHelper'):
            self._batch_size = batch_size
            self._output_dim = output_dim
            self._end_token = tf.tile([0.0], [output_dim * r])

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_dtype(self):
        return tf.int32
    
    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])
    
    def initialize(self, name=None):
        return (tf.tile([False], [self._batch_size]), go_frames(self._batch_size, self._output_dim))

    def sample(self, time, outputs, state, name=None):
        return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        '''Stop on EOS. Otherwise, pass the last output as the next input and pass through state.'''
        with tf.name_scope('TacoTestHelper'):
            finished = tf.reduce_all(tf.equal(outputs, self._end_token), axis=1)
            # Feed last output frame as next input. outputs is [N, output_dim * r]
            next_inputs = outputs[:, -self._output_dim:]
            return (finished, next_inputs, state)


class TacoTrainingHelper(tf.contrib.seq2seq.Helper):
    def __init__(self, targets, output_dim, r):
        # targets is [N, T_out, D]
        with tf.name_scope('TacoTrainingHelper'):
            self._batch_size = tf.shape(targets)[0]
            self._output_dim = output_dim

            # Feed every r-th target frame as input
            self._targets = targets[:, r-1::r, :]

            # Use full length for every target because we don't want to mask the padding frames
            num_steps = tf.shape(self._targets)[1]
            self._lengths = tf.tile([num_steps], [self._batch_size])

    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def sample_ids_dtype(self):
        return tf.int32
    
    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])
    
    def initialize(self, name=None):
        return (tf.tile([False], [self._batch_size]), go_frames(self._batch_size, self._output_dim))

    def sample(self, time, outputs, state, name=None):
        return tf.tile([0], [self._batch_size])  # Return all 0; we ignore them

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        with tf.name_scope(name or 'TacoTrainingHelper'):
            finished = (time + 1 >= self._lengths)
            next_inputs = self._targets[:, time, :]
            return (finished, next_inputs, state)

def go_frames(batch_size, output_dim):
  '''Returns all-zero <GO> frames for a given batch size and output dimension'''
  return tf.tile([[0.0]], [batch_size, output_dim])
