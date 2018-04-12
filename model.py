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
            linear_targets=None,
            audio_lengths=None):

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
            with tf.variable_scope('loss'):
                mel_loss = tf.abs(mel_targets - self.decoder.mel_outputs)

                l1 = tf.abs(linear_targets - self.decoder.linear_outputs)

                self.linear_loss = tf.reduce_mean(l1)
                self.mel_loss = tf.reduce_mean(mel_loss)
                self.loss = self.linear_loss + self.mel_loss

            with tf.variable_scope('optimizer'):
                self.global_step = tf.get_variable("global_step", shape=[], trainable=False,
                              initializer=tf.zeros_initializer, dtype=tf.int32)

                step = tf.cast(self.global_step + 1, dtype=tf.float32)

                self.learning_rate = hyperparams.initial_learning_rate * \
                            tf.train.exponential_decay(1., step, 3000, 0.95)

                optimizer = tf.train.AdamOptimizer(self.learning_rate, hyperparams.adam_beta1, hyperparams.adam_beta2)
                self.gradients, variables = zip(*optimizer.compute_gradients(self.loss))
                clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, 1.0)

                # Memo the total length of audio this model was trained on
                self.total_length = tf.get_variable('total_train_length', [],
                                    initializer=tf.zeros_initializer, dtype=tf.float32)

                update_total_length = tf.assign_add(self.total_length, tf.reduce_sum(audio_lengths))

                # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
                # https://github.com/tensorflow/tensorflow/issues/1122
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS) + [update_total_length]):
                    self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
                        global_step=self.global_step)

                # Logging
                self.training_summary = tf.summary.merge([
                    tf.summary.scalar("total_loss", self.loss),
                    tf.summary.scalar("mel_loss", self.mel_loss),
                    tf.summary.scalar("linear_loss", self.linear_loss),
                    tf.summary.scalar("total_length_hours", self.total_length / 3600),
                    tf.summary.image("alignment_matrix", tf.expand_dims(self.decoder.alignments, 3))
                ])

                self.validation_summary = tf.summary.merge([
                    tf.summary.scalar("validation_total_loss", self.loss),
                    tf.summary.scalar("validation_mel_loss", self.mel_loss),
                    tf.summary.scalar("validation_linear_loss", self.linear_loss),
                    tf.summary.image("v_alignment_matrix", tf.expand_dims(self.decoder.alignments, 3))
                ])

