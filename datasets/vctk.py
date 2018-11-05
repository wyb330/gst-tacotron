from concurrent.futures import ProcessPoolExecutor
from functools import partial
import glob
import librosa
import numpy as np
import os
import re
from util import audio
from util.audio import mulaw_quantize, mulaw, is_mulaw, is_mulaw_quantize

_min_samples = 2000
_threshold_db = 25
_speaker_re = re.compile(r'p([0-9]+)_')


def build_from_path(hparams, input_dirs, out_dir, n_jobs=8, tqdm=lambda x: x):
    wav_paths = glob.glob('%s/wav48/p*/*.wav' % input_dirs)
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures = []
    index = 1
    for wav_path in wav_paths:
        text_path = wav_path.replace('wav48', 'txt').replace('wav', 'txt')
        if os.path.isfile(text_path):
            with open(text_path, 'r') as f:
                text = f.read().strip()
            futures.append(executor.submit(
                partial(_process_utterance, out_dir, index, wav_path, text, hparams)))
        index += 1
    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text, hparams):
    wav = _trim_wav(audio.load_wav(wav_path))
    # Mu-law quantize
    if is_mulaw_quantize(hparams.input_type):
        # [0, quantize_channels)
        out = mulaw_quantize(wav, hparams.quantize_channels)

        # Trim silences
        start, end = audio.start_and_end_indices(out, hparams.silence_threshold)
        wav = wav[start: end]
        out = out[start: end]

        constant_values = mulaw_quantize(0, hparams.quantize_channels)
        out_dtype = np.int16

    elif is_mulaw(hparams.input_type):
        # [-1, 1]
        out = mulaw(wav, hparams.quantize_channels)
        constant_values = mulaw(0., hparams.quantize_channels)
        out_dtype = np.float32

    else:
        # [-1, 1]
        out = wav
        constant_values = 0.
        out_dtype = np.float32

    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

    name = os.path.splitext(os.path.basename(wav_path))[0]
    speaker_id = _speaker_re.match(name).group(1)

    mel_frames = mel_spectrogram.shape[1]

    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        return None

    # Compute the linear scale spectrogram from the wav
    linear_spectrogram = audio.linearspectrogram(wav, hparams).astype(np.float32)
    linear_frames = linear_spectrogram.shape[1]

    # sanity check
    assert linear_frames == mel_frames

    # Ensure time resolution adjustement between audio and mel-spectrogram
    fft_size = hparams.n_fft if hparams.win_size is None else hparams.win_size
    l, r = audio.pad_lr(wav, fft_size, audio.get_hop_size(hparams))

    # Zero pad for quantized signal
    out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
    # print(len(out), mel_frames, audio.get_hop_size(hparams))
    assert len(out) >= mel_frames * audio.get_hop_size(hparams)

    # time resolution adjustement
    # ensure length of raw audio is multiple of hop size so that we can use
    # transposed convolution to upsample
    out = out[:mel_frames * audio.get_hop_size(hparams)]
    assert len(out) % audio.get_hop_size(hparams) == 0
    # time_steps = len(out)

    # Write the spectrogram and audio to disk
    audio_filename = 'vctk-audio-{:05d}.npy'.format(index)
    mel_filename = 'vctk-mel-{:05d}.npy'.format(index)
    # np.save(os.path.join(out_dir, audio_filename), out.astype(out_dtype), allow_pickle=False)
    np.save(os.path.join(out_dir, audio_filename), linear_spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
    return (audio_filename, mel_filename, mel_frames, text)


def _trim_wav(wav):
    '''Trims silence from the ends of the wav'''
    splits = librosa.effects.split(wav, _threshold_db, frame_length=1024, hop_length=512)
    return wav[_find_start(splits):_find_end(splits, len(wav))]


def _find_start(splits):
    for split_start, split_end in splits:
        if split_end - split_start > _min_samples:
            return max(0, split_start - _min_samples)
    return 0


def _find_end(splits, num_samples):
    for split_start, split_end in reversed(splits):
        if split_end - split_start > _min_samples:
            return min(num_samples, split_end + _min_samples)
    return num_samples
