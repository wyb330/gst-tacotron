# Korean voice dataset(Kaldi-based Korean ASR open-source project)
# https://github.com/goodatlas/zeroth
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import re
from util import audio
import librosa

_min_samples = 2000
_threshold_db = 25
_speaker_re = re.compile(r'p([0-9]+)_')


def build_from_path(hparams, input_dirs, out_dir, n_jobs=8, tqdm=lambda x: x):
    dir_list = os.listdir(input_dirs)
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures = []
    index = 1
    text_file = os.path.join(input_dirs, 'trans.txt')
    for d in dir_list:
        path = os.path.join(input_dirs, d)
        if os.path.isfile(text_file):
            with open(text_file, 'r', encoding='utf8') as f:
                for line in f:
                    parts = line.strip().split()
                    wav = os.path.join(path, '%s_%s.wav' % (d, parts[0]))
                    if os.path.exists(wav):
                        text = ' '.join(parts[1:])
                        futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav, text, hparams)))
                        index += 1
    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text, hparams):
    '''Preprocesses a single utterance audio/text pair.

    This writes the mel and linear scale spectrograms to disk and returns a tuple to write
    to the train.txt file.

    Args:
      out_dir: The directory to write the spectrograms into
      index: The numeric index to use in the spectrogram filenames.
      wav_path: Path to the audio file containing the speech input
      text: The text spoken in the input audio file

    Returns:
      A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
    '''

    # Load the audio to a numpy array:
    wav = _trim_wav(audio.load_wav(wav_path))

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        return None

    # Write the spectrograms to disk:
    spectrogram_filename = 'seoul-spec-%05d.npy' % index
    mel_filename = 'seoul-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example:
    return (spectrogram_filename, mel_filename, n_frames, text)


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
