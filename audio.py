import numpy as np
import librosa
import scipy

def spectrogram(hyperparams, audio):
    # Truncate excessively long samples
    audio = audio[:hyperparams.max_audio_length]

    # Emphasize high-frequency
    preemp = preemphasis(audio, hyperparams.preemphasis)

    # Obtain STFT spectra
    n_fft, hop_length, win_length = get_stft_params(hyperparams)
    spectra = stft(preemp, n_fft, hop_length, win_length)
    mag = np.abs(spectra)

    # Truncate to 4-frame boundary
    l = mag.shape[1] - mag.shape[1] % hyperparams.reduction_factor
    mag = mag[:, :l]

    # Linear spectrum
    linear = amp_to_db(mag) - hyperparams.ref_level_db
    linear = normalize(linear, hyperparams.min_level_db)

    # Mel spectrum
    mel = amp_to_db(linear_to_mel(mag, n_fft, hyperparams.sample_rate, hyperparams.num_mels))
    mel = normalize(mel, hyperparams.min_level_db)

    return mel, linear

mel_basis = None
def get_mel_basis(n_fft, sample_rate, num_mels):
    global mel_basis
    if mel_basis is None:
        mel_basis = librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels).astype(np.float32)
    return mel_basis

def linear_to_mel(spectrogram, n_fft, sample_rate, num_mels):
    mel_basis = get_mel_basis(n_fft, sample_rate, num_mels)
    return np.dot(mel_basis, spectrogram)

inv_mel_basis = None
def mel_to_linear(mel_spectrogram, n_fft, sample_rate, num_mels):
    global inv_mel_basis
    if inv_mel_basis is None:
        inv_mel_basis = np.linalg.pinv(get_mel_basis(n_fft, sample_rate, num_mels))
    return np.maximum(1e-10, np.dot(inv_mel_basis, mel_spectrogram))

def normalize(x, min_level):
    return np.clip((x - min_level) / -min_level, 0, 1)

def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def get_stft_params(hyperparams):
    n_fft = (hyperparams.num_freq - 1) * 2
    hop_length = int(hyperparams.frame_shift_ms / 1000 * hyperparams.sample_rate)
    win_length = int(hyperparams.frame_length_ms / 1000 * hyperparams.sample_rate)
    return n_fft, hop_length, win_length

def stft(x, n_fft, hop_length, win_length):
    return librosa.stft(
        y=x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length)

def preemphasis(x, coeff):
    return x[1:] - coeff * x[:-1]

def reconstruct(hyperparams, spectrogram, from_mel=False):
    n_fft, hop_length, win_length = get_stft_params(hyperparams)

    def stft(y):
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    def istft(y):
        return librosa.istft(y, hop_length=hop_length, win_length=win_length)

    def db_to_amp(x):
        return np.power(10.0, x * 0.05)

    def inv_preemphasis(x):
        return scipy.signal.lfilter([1], [1, -hyperparams.preemphasis], x)

    def denormalize(S):
        return (np.clip(S, 0, 1) * -hyperparams.min_level_db) + hyperparams.min_level_db

    if from_mel:
        S = db_to_amp(denormalize(spectrogram))
        S = mel_to_linear(S, n_fft, hyperparams.sample_rate, hyperparams.num_mels)
    else:
        S = db_to_amp(denormalize(spectrogram) + hyperparams.ref_level_db)

    S = S ** hyperparams.power

    # Reconstruct phase (Griffin-Lim reconstruction)
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)

    y = istft(S_complex * angles)
    for i in range(hyperparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(stft(y)))
        y = istft(S_complex * angles)

    # unapply preemphasis
    return inv_preemphasis(y)

def read_audio(path, sr):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav

# Write audio to file
def write_audio(path, y, sr):
    librosa.output.write_wav(path, y, sr, norm=True)
