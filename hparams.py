import tensorflow as tf

# Default hyperparameters:
hparams = tf.contrib.training.HParams(
    # Comma-separated list of cleaners to run on text prior to training and eval. For non-English
    # text, you may want to use "basic_cleaners" or "transliteration_cleaners" See TRAINING_DATA.md.
    cleaners='english_cleaners',

    # Audio:
    num_mels=80,
    num_freq=513,  # (= n_fft / 2 + 1) only used when adding linear spectrograms post processing network
    sample_rate=22050,
    frame_length_ms=50,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,
    clip_mels_length=False,  # For cases of OOM (Not really recommended, working on a workaround)

    # Mel spectrogram
    n_fft=1024,  # Extra window size is filled with 0 paddings to match this parameter
    hop_size=256,  # For 22050Hz, 275 ~= 12.5 ms
    win_size=None,  # For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft)
    frame_shift_ms=12.5,

    # Model:
    outputs_per_step=2,
    embed_depth=256,
    prenet_depths=[256, 128],
    encoder_depth=256,
    rnn_depth=256,

    # Attention
    attention_depth=256,

    # Training:
    batch_size=32,
    adam_beta1=0.9,
    adam_beta2=0.999,
    initial_learning_rate=0.002,
    decay_learning_rate=True,
    use_cmudict=False,  # Use CMUDict during training to learn pronunciation of ARPAbet phonemes

    # Eval:
    max_iters=1000,
    griffin_lim_iters=60,
    power=1.5,  # Power to raise magnitudes to prior to Griffin-Lim

    # Global style token
    use_gst=True,
    # When false, the scripit will do as the paper  "Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron"
    num_gst=10,
    num_heads=4,  # Head number for multi-head attention
    style_embed_depth=256,
    reference_filters=[32, 32, 64, 64, 128, 128],
    reference_depth=128,
    style_att_type="mlp_attention",  # Attention type for style attention module (dot_attention, mlp_attention)
    style_att_dim=128,

    input_type="raw",
    max_mel_frames=900,  # Only relevant when clip_mels_length = True
    # Mel and Linear spectrograms normalization/scaling and clipping
    signal_normalization=True,
    allow_clipping_in_normalization=True,  # Only relevant if mel_normalization = True
    symmetric_mels=True,  # Whether to scale the data to be symmetric around 0
    max_abs_value=4.,  # max absolute value of data. If symmetric, data will be [-max, max] else [0, max]

)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
