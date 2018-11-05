# Korean voice dataset(Kaldi-based Korean ASR open-source project)
# https://github.com/goodatlas/zeroth
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import glob
import numpy as np
import os
import re
from util import audio

_min_samples = 2000
_threshold_db = 25
_speaker_re = re.compile(r'p([0-9]+)_')


def build_from_path(hparams, input_dirs, out_dir, n_jobs=8, tqdm=lambda x: x):
    dir_list = os.listdir('%s/train_data_01/003' % input_dirs)
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures = []
    index = 1
    for d in dir_list:
        path = os.path.join('%s/train_data_01/003' % input_dirs, d)
        text_file = glob.glob('{}/*.txt'.format(path))
        if os.path.isfile(text_file[0]):
            with open(text_file[0], 'r', encoding='utf8') as f:
                for line in f:
                    parts = line.strip().split()
                    wav = os.path.join(path, '%s.flac' % parts[0])
                    text = ' '.join(parts[1:])
                    futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav, text)))
                    index += 1
    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text):
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
    wav = audio.load_wav(wav_path)

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

    # Write the spectrograms to disk:
    spectrogram_filename = 'zeroth-spec-%05d.npy' % index
    mel_filename = 'zzeroth-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example:
    return (spectrogram_filename, mel_filename, n_frames, text)
