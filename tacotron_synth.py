import tensorflow as tf
import numpy as np
import model
import audio
import text
from hyperparams import hyperparams
import matplotlib.pyplot as plt

def synth():
    tokens_ph = tf.placeholder(tf.int32, [None], "tokens_ph")

    tokens_op = tf.expand_dims(tokens_ph, 0)
    token_lengths = tf.expand_dims(tf.shape(tokens_ph)[0], 0)

    tacotron = model.Model(
        hyperparams,
        is_training=False,
        inputs=tokens_op,
        input_lengths=token_lengths)

    melspectrum_op = tacotron.decoder.mel_outputs
    spectrum_op = tacotron.decoder.linear_outputs
    alignments_op = tacotron.decoder.alignments

    saver = tf.train.Saver()

    with tf.Session() as sess:

        restore_path = input('Restore path: ')
        saver.restore(sess, restore_path)

        while True:
            sentence = input('Input: ')
            if sentence == '':
                sentence = "In the beginning God created the heavens and the earth."
            tokens = text.encode(sentence)
            melspectrum, spectrum, alignments = sess.run([melspectrum_op, spectrum_op, alignments_op], {tokens_ph: tokens})
            plt.figure()
            plt.imshow(melspectrum[0])
            plt.figure()
            plt.imshow(spectrum[0])
            plt.figure()
            plt.imshow(alignments[0])
            plt.show()
            signal = audio.reconstruct(hyperparams, spectrum[0].T)
            audio.write_audio(sentence + ".wav", signal, hyperparams.sample_rate)

if __name__ == '__main__':
    synth()
