import tensorflow as tf
import modules

class Decoder:
    def __init__(self, hyperparams, is_training, encoder_outputs, mel_targets=None):
        # mel_targets: (batch, max_sample_length, num_mels)
        # encoder_outputs: (batch, max_sentence_length, enc_rnn_size * 2)

        batch_size = tf.shape(encoder_outputs)[0]

        #GRU = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell
        GRU = tf.contrib.rnn.GRUCell

        dec_prenet = modules.DecoderPrenetWrapper(
                GRU(hyperparams.attention_state_size),
                is_training,
                hyperparams.dec_prenet_sizes,
                hyperparams.dropout_prob)

        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                hyperparams.attention_size, encoder_outputs,
                normalize=True)

        attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                dec_prenet,
                attention_mechanism,
                alignment_history=True,
                output_attention=False)

        # Concatenate attention context vector and RNN cell output into a 512D vector.
        # [N, T_in, attention_size+attention_state_size]
        concat_cell = modules.ConcatOutputAndAttentionWrapper(attention_cell)

        # Synthesis model for inference
        cells = [concat_cell]
        for layer_index in range(hyperparams.dec_layer_num):
            cell = GRU(hyperparams.dec_rnn_size)
            if layer_index == 0:
                cells.append(cell)
            else:
                cells.append(tf.contrib.rnn.ResidualWrapper(cell))

        # [N, T_in, 256]
        decoder_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # GRU layers + linear projection
        # Weights
        proj_input_size = hyperparams.dec_rnn_size
        proj_output_size = hyperparams.num_mels * hyperparams.reduction_factor
        decoder_proj_weights = tf.get_variable(
            'decoder_proj_weights',
            shape=[proj_input_size, proj_output_size],
            initializer=tf.contrib.layers.xavier_initializer())

        if is_training:
            # Training Model for speed
            r = hyperparams.reduction_factor
            pre_padded_mel = tf.pad(mel_targets[:, r-1:-r+1:r], [[0,0], [1,0], [0,0]])
            gru_outputs, states = tf.nn.dynamic_rnn(
                decoder_cell,
                pre_padded_mel,
                dtype=tf.float32,
                swap_memory=True,
                scope='rnn')

            decoder_outputs = tf.matmul(
                tf.reshape(gru_outputs, (-1, hyperparams.dec_rnn_size)),
                decoder_proj_weights)

            # Grab alignments (N, T_out, T_in)
            self.alignments = tf.transpose(states[0].alignment_history.stack(), (1, 0, 2))

        else:
            proj_decoder_cell = modules.OutputProjectionWrapper(decoder_cell, decoder_proj_weights)

            # Synthesis model for inference
            helper = modules.TacoTestHelper(
                    batch_size,
                    hyperparams.num_mels,
                    hyperparams.reduction_factor)

            decoder_init_state = proj_decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            (decoder_outputs, _), self.final_state, _ = \
                    tf.contrib.seq2seq.dynamic_decode(
                            tf.contrib.seq2seq.BasicDecoder(proj_decoder_cell, helper, decoder_init_state),
                            maximum_iterations=hyperparams.max_iters,
                            swap_memory=True,
                            scope='rnn')

            # Grab alignments from the final decoder state:
            self.alignments = tf.transpose(self.final_state[0].alignment_history.stack(), (1, 0, 2))

        # [N, T_out, M]
        self.mel_outputs = tf.reshape(decoder_outputs, [batch_size, -1, hyperparams.num_mels])

        # Add post-processing CBHG:
        # [N, T_out, 256]
        post_outputs = modules.cbhg(
                self.mel_outputs, None, is_training,
                hyperparams.post_bank_size,
                hyperparams.post_bank_channel_size,
                hyperparams.post_maxpool_width,
                hyperparams.post_highway_depth,
                hyperparams.post_rnn_size,
                hyperparams.post_proj_sizes,
                hyperparams.post_proj_width,
                scope='post_cbhg')

        self.linear_outputs = tf.layers.dense(post_outputs, hyperparams.num_freq) # [N, T_out, F]

