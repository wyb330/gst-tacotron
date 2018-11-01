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

    tacotron_zoneout_rate=0.1,  # zoneout rate for all LSTM cells in the network
    tacotron_dropout_rate=0.5,  # dropout rate for all convolutional layers + prenet

    prenet_layers=[256, 256],  # number of layers and number of units of prenet
    decoder_layers=2,  # number of decoder lstm layers
    decoder_lstm_units=1024,  # number of decoder lstm units on each layer

    postnet_num_layers=5,  # number of postnet convolutional layers
    postnet_kernel_size=(5,),  # size of postnet convolution filters for each layer
    postnet_channels=512,  # number of postnet convolution filters for each layer

    enc_conv_num_layers=3,  # number of encoder convolutional layers
    enc_conv_kernel_size=(5,),  # size of encoder convolution filters for each layer
    enc_conv_channels=512,  # number of encoder convolutions filters for each layer
    encoder_lstm_units=256,  # number of lstm units for each direction (forward and backward)

    # Decoder RNN learning can take be done in one of two ways:
    #	Teacher Forcing: vanilla teacher forcing (usually with ratio = 1). mode='constant'
    #	Curriculum Learning Scheme: From Teacher-Forcing to sampling from previous outputs is function of global step. (teacher forcing ratio decay) mode='scheduled'
    # The second approach is inspired by:
    # Bengio et al. 2015: Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.
    # Can be found under: https://arxiv.org/pdf/1506.03099.pdf
    tacotron_teacher_forcing_mode='constant',
    # Can be ('constant' or 'scheduled'). 'scheduled' mode applies a cosine teacher forcing ratio decay. (Preference: scheduled)
    tacotron_teacher_forcing_ratio=1.,
    # Value from [0., 1.], 0.=0%, 1.=100%, determines the % of times we force next decoder inputs, Only relevant if mode='constant'
    tacotron_teacher_forcing_init_ratio=1.,  # initial teacher forcing ratio. Relevant if mode='scheduled'
    tacotron_teacher_forcing_final_ratio=0.,  # final teacher forcing ratio. Relevant if mode='scheduled'
    tacotron_teacher_forcing_start_decay=10000,
    # starting point of teacher forcing ratio decay. Relevant if mode='scheduled'
    tacotron_teacher_forcing_decay_steps=280000,
    # Determines the teacher forcing ratio decay slope. Relevant if mode='scheduled'
    tacotron_teacher_forcing_decay_alpha=0.,  # teacher forcing ratio decay rate. Relevant if mode='scheduled'
    ###########################################################################################################################################
    attention_filters=32,  # number of attention convolution filters
    attention_kernel=(31,),  # kernel size of attention convolution
    cumulative_weights=True,
    tacotron_reg_weight=1e-6,  # regularization weight (for L2 regularization)
    tacotron_scale_regularization=True,

    tacotron_decay_learning_rate=True,  # boolean, determines if the learning rate will follow an exponential decay
    tacotron_start_decay=50000,  # Step at which learning decay starts
    tacotron_decay_steps=40000,  # Determines the learning rate decay slope (UNDER TEST)
    tacotron_decay_rate=0.2,  # learning rate decay rate (UNDER TEST)
    tacotron_initial_learning_rate=1e-3,  # starting learning rate
    tacotron_final_learning_rate=1e-5,  # minimal learning rate

    tacotron_adam_beta1=0.9,  # AdamOptimizer beta1 parameter
    tacotron_adam_beta2=0.999,  # AdamOptimizer beta2 parameter
    tacotron_adam_epsilon=1e-6,  # AdamOptimizer beta3 parameter

)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
