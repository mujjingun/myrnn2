import tensorflow as tf
import numpy as np
import model
import audio
import text
from hyperparams import hyperparams

import os, time

def parse_line(line):
    # Parse line from csv
    filename, sentence, duration = line.decode('ascii').split('\t')

    # Audio file
    wav_path = os.path.join(hyperparams.dataset_path, filename + '.wav')
    wave = audio.read_audio(wav_path, hyperparams.sample_rate)
    audio_length = wave.shape[0] / hyperparams.sample_rate

    # Calculate spectrum
    mel, linear = audio.spectrogram(hyperparams, wave)

    # Encode sentence
    tokens = text.encode(sentence)

    return mel.T, linear.T, tokens, np.int32(tokens.size), np.float32(audio_length)

def train():
    # Read file
    transcript = os.path.join(hyperparams.dataset_path, 'transcript.txt')
    lines = tf.data.TextLineDataset(transcript)

    # Datasets
    data_shapes = ([None, hyperparams.num_mels], [None, hyperparams.num_freq], [None], [], [])

    map_func = lambda line: tf.py_func(parse_line, [line], [tf.float32, tf.float32, tf.int32, tf.int32, tf.float32])

    validation_dataset = lines.map(map_func) \
        .padded_batch(
            hyperparams.validation_batch_size,
            data_shapes
        ).take(1).repeat()
    validation_iter_handle_op = validation_dataset.make_one_shot_iterator().string_handle()

    training_dataset = lines.prefetch(hyperparams.prefetch_size) \
        .shuffle(hyperparams.shuffle_size) \
        .repeat() \
        .map(map_func) \
        .padded_batch(
            hyperparams.batch_size,
            data_shapes
        )
    training_iter_handle_op = training_dataset.make_one_shot_iterator().string_handle()

    # Iterator
    handle_ph = training_iter_handle_op
    iterator = tf.data.Iterator.from_string_handle(
        handle_ph, training_dataset.output_types, training_dataset.output_shapes)

    mel_op, linear_op, tokens_op, token_lengths_op, audio_lengths_op = iterator.get_next()

    # Construct graph
    tacotron = model.Model(
        hyperparams,
        is_training=True,
        inputs=tokens_op,
        input_lengths=token_lengths_op,
        mel_targets=mel_op,
        linear_targets=linear_op,
        audio_lengths=audio_lengths_op)

    config = tf.ConfigProto(
    #    allow_soft_placement=True,
    #    log_device_placement=True
    )

    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(hyperparams.summary_path, sess.graph)

        restore_path = input('Restore path (leave blank for new training): ')
        if restore_path != '':
            saver.restore(sess, restore_path)
            iteration = sess.run(tacotron.global_step)
            print("Resuming training from iteration {}.".format(iteration))
        else:
            sess.run(tf.global_variables_initializer())

        # Handle to pass when doing validation runs
        validation_handle = sess.run(validation_iter_handle_op)

        while True:
            # Evaluate validation data
            summary, iteration, total_length, alignments = sess.run(
                [tacotron.validation_summary, tacotron.global_step, tacotron.total_length, tacotron.decoder.alignments],
                {handle_ph: validation_handle})

            print(np.sum(alignments[0], axis=1))

            train_writer.add_summary(summary, iteration)
            print('Processed {} hours of data.'.format(total_length / 3600))

            save_path = saver.save(sess, hyperparams.save_path, iteration)
            print('Saved to', save_path)

            # Train for some steps
            for _ in range(hyperparams.validate_every_n_steps):
                #start_time = time.time()
                summary, _, iteration, loss = sess.run(
                    [tacotron.training_summary, tacotron.optimize, tacotron.global_step, tacotron.loss])
                train_writer.add_summary(summary, iteration)

                if iteration % 10 == 0:
                    print("{}: {}".format(iteration, loss))
                #print((time.time() - start_time) / hyperparams.batch_size, "seconds per data point")

if __name__ == "__main__":
    train()
