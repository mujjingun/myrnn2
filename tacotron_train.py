import tensorflow as tf
import model
import audio
from hyperparams import hyperparams

import os, time

def parse_line(line):
    with tf.device('/cpu:0'):
        # Parse line from csv
        filename, sentence, duration = tf.decode_csv(
            line,
            record_defaults=[tf.constant([], dtype=tf.string), tf.constant([], dtype=tf.string), []],
            field_delim=',')

        # Audio file
        wav_path = tf.string_join([hyperparams.dataset_path, "/", filename])
        wav_binary = tf.read_file(wav_path)
        wave = tf.contrib.ffmpeg.decode_audio(
            wav_binary,
            file_format='wav',
            channel_count=1,
            samples_per_second=hyperparams.sample_rate)[:, 0] # first channel
        audio_length = tf.cast(tf.shape(wave)[0], tf.float32) / hyperparams.sample_rate

        # Calculate spectrum
        mel, linear = audio.spectrogram(hyperparams, wave)

        # Encode sentence
        tokens = tf.cast(tf.decode_raw(sentence, tf.int8), tf.int32)
        tokens = tf.nn.relu(tokens - 96) + 1 # EOS = 0, ' ' = 1, 'a' = 2, 'b' = 3, ...
        tokens = tf.concat([tokens, [0]], axis=0) # Append EOS symbol

        return mel, linear, tokens, tf.shape(tokens)[0], audio_length

def train():
    # Read file
    transcript = os.path.join(hyperparams.dataset_path, 'alignment.csv')
    lines = tf.data.TextLineDataset(transcript)

    # Datasets
    data_shapes = ([None, hyperparams.num_mels], [None, hyperparams.num_freq], [None], [], [])

    validation_dataset = lines.map(parse_line).take(hyperparams.validation_batch_size) \
        .repeat().padded_batch(
            hyperparams.validation_batch_size,
            data_shapes
        )
    validation_iter_handle_op = validation_dataset.make_one_shot_iterator().string_handle()

    training_dataset = lines.prefetch(hyperparams.prefetch_size) \
        .map(parse_line) \
        .repeat() \
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
        allow_soft_placement=True,
    #    log_device_placement=True
    )

    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        restore_path = input('Restore path (leave blank for new training): ')
        if restore_path != '':
            saver.restore(sess, restore_path)
            iteration = sess.run(tacotron.global_step)
            print("Resuming training from iteration {}.".format(iteration))
        else:
            sess.run(tf.global_variables_initializer())

        validation_handle = sess.run(validation_iter_handle_op)

        while True:
            # Evaluate validation data
            validation_loss, iteration = sess.run(
                [tacotron.loss, tacotron.global_step],
                {handle_ph: validation_handle})
            print("Validation loss:", validation_loss)

            save_path = saver.save(sess, hyperparams.save_path, iteration)
            print('Saved to', save_path)

            # Run training for some steps
            for _ in range(hyperparams.validate_every_n_steps):
                start_time = time.time()
                _, loss, total_length = sess.run(
                    [tacotron.optimize, tacotron.loss, tacotron.total_length])
                print("loss: {} for {} hours of data".format(loss, total_length / 3600))
                print((time.time() - start_time) / hyperparams.batch_size, "seconds per data point")

if __name__ == "__main__":
    train()
