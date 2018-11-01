import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_alignment(alignment, path, info=None):
    fig, ax = plt.subplots()
    im = ax.imshow(
        alignment,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig(path, format='png')


def split_title_line(title_text, max_words=5):
    """
    A function that splits any string based on specific character
    (returning it with the string), with maximum number of words on it
    """
    seq = title_text.split()
    return '\n'.join([' '.join(seq[i:i + max_words]) for i in range(0, len(seq), max_words)])


def plot_spectrogram(pred_spectrogram,
                     path, info=None,
                     split_title=False,
                     target_spectrogram=None,
                     max_len=None,
                     head=None):
    if max_len is not None:
        target_spectrogram = target_spectrogram[:max_len]
        pred_spectrogram = pred_spectrogram[:max_len]

    if info is not None:
        if split_title:
            title = split_title_line(info)
        else:
            title = info

    fig = plt.figure(figsize=(10, 8))
    # Set common labels
    fig.text(0.5, 0.18, title, horizontalalignment='center', fontsize=16)

    # target spectrogram subplot
    if target_spectrogram is not None:
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)

        im = ax1.imshow(np.rot90(target_spectrogram), interpolation='none')
        ax1.set_title('Target Mel-Spectrogram')
        fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax1)
        ax2.set_title('Predicted Mel-Spectrogram')
    else:
        ax2 = fig.add_subplot(211)

    im = ax2.imshow(np.rot90(pred_spectrogram), interpolation='none')
    fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax2)

    if head:
        plt.title(head)
    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close()
