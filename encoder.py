import tensorflow as tf
import modules

class Encoder:
    def __init__(self, hyperparams, is_training, inputs, input_lengths):
        # inputs: (batch, max_input_length)
        # input_lengths: (batch)
    
        # Embeddings
        char_embed_table = tf.get_variable(
                'embedding', 
                [hyperparams.num_symbols, hyperparams.embedding_size],
                dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.5))
        
        # [N, T_in, embedding_size]
        char_embedded_inputs = tf.nn.embedding_lookup(char_embed_table, inputs)

        # [N, T_in, enc_prenet_sizes[-1]]
        prenet_outputs = modules.prenet(
                char_embedded_inputs,
                is_training,
                layer_sizes = hyperparams.enc_prenet_sizes,
                drop_prob   = hyperparams.dropout_prob,
                scope='prenet')

        encoder_outputs = modules.cbhg(
                prenet_outputs, input_lengths, is_training,
                hyperparams.enc_bank_size,
                hyperparams.enc_bank_channel_size,
                hyperparams.enc_maxpool_width,
                hyperparams.enc_highway_depth,
                hyperparams.enc_rnn_size,
                hyperparams.enc_proj_sizes,
                hyperparams.enc_proj_width,
                scope="encoder_cbhg")
        
        self.encoder_outputs = encoder_outputs
