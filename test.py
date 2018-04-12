import audio
import text
from hyperparams import hyperparams

import matplotlib.pyplot as plt

import os, time

def test_audio_conv(audio_path):
    # Audio file
    wav_path = os.path.join(hyperparams.dataset_path, filename + '.wav')
    wave = audio.read_audio(wav_path, hyperparams.sample_rate)
    audio_length = wave.shape[0] / hyperparams.sample_rate

    # Calculate spectrum
    mel, linear = audio.spectrogram(hyperparams, wave)

    #plt.imshow(mel)
    from_mel = audio.mel_to_linear(mel, (hyperparams.num_freq - 1) * 2, hyperparams.sample_rate, hyperparams.num_mels)
    plt.imshow(from_mel)
    plt.show()
    plt.imshow(linear)
    plt.show()

    signal = audio.reconstruct(hyperparams, linear)
    audio.write_audio('test.wav', signal, hyperparams.sample_rate)

    signal = audio.reconstruct(hyperparams, mel, from_mel=True)
    audio.write_audio('test_mel.wav', signal, hyperparams.sample_rate)

test_audio_conv('Genesis/Genesis_1-1')
