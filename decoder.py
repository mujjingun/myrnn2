import tensorflow as tf
import modules

class Decoder:
    def __init__(self, hyperparams, is_training, encoder_outputs, mel_targets=None):
        # mel_targets: (batch, max_sample_length, num_mels)
        # encoder_outputs: (batch, max_sentence_length, enc_rnn_size * 2)
        
        batch_size = tf.shape(encoder_outputs)[0]
    
        dec_prenet = modules.DecoderPrenetWrapper(
                tf.contrib.rnn.GRUCell(hyperparams.attention_state_size),
                is_training, 
                hyperparams.dec_prenet_sizes, 
                hyperparams.dropout_prob)
        
        attention_mechanism = tf.contrib.seq2seq.BahdanauMonotonicAttention(
                hyperparams.attention_size, encoder_outputs)
        
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                dec_prenet,
                attention_mechanism,
                alignment_history=True,
                output_attention=False
        )
        
        # Concatenate attention context vector and RNN cell output into a 512D vector.
        # [N, T_in, attention_size+attention_state_size]
        concat_cell = modules.ConcatOutputAndAttentionWrapper(attention_cell)
        
        # Decoder (layers specified bottom to top):
        cells = [tf.contrib.rnn.OutputProjectionWrapper(concat_cell, hyperparams.dec_rnn_size)]
        for _ in range(hyperparams.dec_layer_num):
            cells.append(
                tf.contrib.rnn.ResidualWrapper(
                    tf.contrib.rnn.GRUCell(hyperparams.dec_rnn_size)
                )
            )

        # [N, T_in, 256]
        decoder_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # Project onto r mel spectrograms (predict r outputs at each RNN step):
        output_cell = tf.contrib.rnn.OutputProjectionWrapper(
                decoder_cell, hyperparams.num_mels * hyperparams.reduction_factor)
    
        if is_training:
            helper = modules.TacoTrainingHelper(
                    mel_targets, hyperparams.num_mels, hyperparams.reduction_factor)
            max_iters = None
        else:
            helper = modules.TacoTestHelper(
                    batch_size, hyperparams.num_mels, hyperparams.reduction_factor)
            max_iters = hyperparams.max_iters

        decoder_init_state = output_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        
        (decoder_outputs, _), self.final_decoder_state, _ = \
                tf.contrib.seq2seq.dynamic_decode(
                        tf.contrib.seq2seq.BasicDecoder(output_cell, helper, decoder_init_state),
                        maximum_iterations=max_iters,
                        swap_memory=True)
        
        # [N, T_out, M]
        self.mel_outputs = tf.reshape(
                decoder_outputs, [batch_size, -1, hyperparams.num_mels])

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

        # Grab alignments from the final decoder state:
        self.alignments = tf.transpose(
                self.final_decoder_state[0].alignment_history.stack(), [1, 2, 0])
