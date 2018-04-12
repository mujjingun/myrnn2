import tensorflow as tf
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

    spectrum_op = tacotron.decoder.linear_outputs

    saver = tf.train.Saver()

    with tf.Session() as sess:

        restore_path = input('Restore path: ')
        saver.restore(sess, restore_path)

        while True:
            sentence = input('Input: ')
            tokens = text.encode(sentence)
            spectrum = sess.run(spectrum_op, {tokens_ph: tokens})
            plt.imshow(spectrum[0])
            plt.show()
            signal = audio.reconstruct(hyperparams, spectrum[0])
            audio.write_audio(sentence + ".wav", signal, hyperparams.sample_rate)

if __name__ == '__main__':
    synth()
