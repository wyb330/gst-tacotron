from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from util import audio
import json


def is_json(file):
    if file.endswith('.json'):
        return True
    else:
        return False


def build_from_path(input_dirs, out_dir, text_file, n_jobs=8, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=n_jobs)
    futures = []
    index = 1
    file = os.path.join(input_dirs, text_file)
    if os.path.exists(file):
        with open(file, 'r', encoding='utf8') as f:
            if is_json(text_file):
                info = json.load(f)
                for wav_path, text in info.items():
                    wav = wav_path
                    futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav, text)))
                    index += 1
            else:
                for line in f:
                    parts = line.strip().split()
                    wav = parts[0]
                    text = ' '.join(parts[1:])
                    futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav, text)))
                    index += 1
    else:
        raise Exception('Text file "{}" not exists.'.format(text_file))

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
    spectrogram_filename = 'voice-spec-%05d.npy' % index
    mel_filename = 'voice-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, spectrogram_filename), spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example:
    return (spectrogram_filename, mel_filename, n_frames, text)

