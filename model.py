import tensorflow as tf
from encoder import Encoder
from decoder import Decoder

class Model:
    def __init__(self, 
            hyperparams,
            is_training,
            inputs, 
            input_lengths,
            mel_targets=None,
            linear_targets=None):
        
        self.encoder = Encoder(
            hyperparams, 
            is_training, 
            inputs, 
            input_lengths)
        
        self.decoder = Decoder(
            hyperparams, 
            is_training, 
            self.encoder.encoder_outputs, 
            mel_targets)
        
        if is_training:
            with tf.variable_scope('loss') as scope:
                mel_loss = tf.abs(mel_targets - self.decoder.mel_outputs)

                l1 = tf.abs(linear_targets - self.decoder.linear_outputs)

                self.linear_loss = tf.reduce_mean(l1)
                self.mel_loss = tf.reduce_mean(mel_loss)
                self.loss = self.linear_loss + self.mel_loss
