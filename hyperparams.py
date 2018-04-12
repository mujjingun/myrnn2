import tensorflow as tf

hyperparams = {
    # Save/Restore model
    'save_path': '../models/tacotron',
    'summary_path': "../summary/tacotron",

    # Training
    'dataset_path': "../../WEB/",
    'shuffle_size': 128,
    'prefetch_size': 128,
    'validation_batch_size': 16,
    'validate_every_n_steps': 100,
    'batch_size': 8,
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'initial_learning_rate': 0.002,

    # Audio
    "sample_rate": 12000,
    "frame_length_ms": 50,
    "frame_shift_ms": 12.5, # frame_length / 4
    "preemphasis": 0.97,
    "min_level_db": -100,
    "ref_level_db": 20,
    "num_freq": 1025,   # for linear output
    "num_mels": 80,     # for mel-frequency output
    "max_audio_length": 120000, # cut off audio at 10 sec

    # String
    "num_symbols": 30,
    "embedding_size": 256,

    # Encoder
    "enc_prenet_sizes": [256, 128],
    "enc_bank_size": 16, # kernel size = 1 .. enc_bank_size
    "enc_bank_channel_size": 128,
    "enc_maxpool_width": 2,
    "enc_highway_depth": 4,
    "enc_rnn_size": 128,
    "enc_proj_sizes": [128, 128],
    "enc_proj_width": 3,

    # Decoder
    "dec_prenet_sizes": [256, 128],
    "dec_rnn_size": 256,
    "dec_layer_num": 2,

    # Postprocessing net
    "post_bank_size": 8,
    "post_bank_channel_size": 128,
    "post_maxpool_width": 2,
    "post_highway_depth": 4,
    "post_rnn_size": 128,
    "post_proj_sizes": [256, 80],
    "post_proj_width": 3,

    # Attention
    "attention_size": 256,
    "attention_state_size": 256,

    # Misc.
    "dropout_prob": 0.5,
    "reduction_factor": 4,

    # Inference
    'min_tokens': 50,
    'min_iters': 30,
    'max_iters': 200,
    'griffin_lim_iters': 60,
    'power': 1.5, # Power to raise magnitudes to prior to Griffin-Lim
}

hyperparams = tf.contrib.training.HParams(**hyperparams)
