import tensorflow as tf
import model
import audio
from hyperparams import hyperparams

import os, time

def parse_line(line):
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
    
    # Calculate spectrum
    mel, linear = audio.spectrogram(hyperparams, wave)
    
    # Encode sentence
    tokens = tf.cast(tf.decode_raw(sentence, tf.int8), tf.int32)
    tokens = tf.nn.relu(tokens - 96) # ' ' = 0, 'a' = 1, 'b' = 2, ...
    tokens = tf.concat([tokens, [hyperparams.num_symbols - 1]], axis=0) # Append EOS symbol
    
    return mel, linear, tokens, tf.shape(tokens)[0]

def train():
    # Read file
    transcript = os.path.join(hyperparams.dataset_path, 'alignment.csv')
    lines = tf.data.TextLineDataset(transcript)
    
    # Dataset
    dataset = lines.map(parse_line)
    #dataset = dataset.shuffle(hyperparams.shuffle_size)
    dataset = dataset.repeat()
    dataset = dataset.padded_batch(
        hyperparams.batch_size,
        ([None, hyperparams.num_mels], [None, hyperparams.num_freq], [None], []))
    
    # Iterator
    iterator = dataset.make_initializable_iterator()
    next_batch_op = iterator.get_next()
    
    # Construct graph
    mel_op, linear_op, tokens_op, token_lengths_op = next_batch_op
    tacotron = model.Model(
        hyperparams,
        is_training=True,
        inputs=tokens_op,
        input_lengths=token_lengths_op,
        mel_targets=mel_op,
        linear_targets=linear_op)

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        sess.run(tf.global_variables_initializer())
        
        while True:
            start_time = time.time()
            sess.run(tacotron.loss)
            print((time.time() - start_time) / hyperparams.batch_size, "seconds per data point")
        
        #signal = audio.reconstruct(hyperparams, linear[0])
        #import matplotlib.pyplot as plt
        #plt.imshow(linear[0])
        #plt.show()
        #print(list(token_lengths))
        #plt.plot(signal)
        #plt.show()
        #audio.write_audio("test.wav", signal, hyperparams.sample_rate)
        #print(mel)
        

if __name__ == "__main__":
    train()
