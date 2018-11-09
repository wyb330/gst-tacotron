import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from datasets import blizzard, ljspeech, blizzard2013, vctk, zeroth, seoul
from hparams import hparams


def preprocess_blizzard(args):
    in_dir = os.path.join(args.base_dir, 'Blizzard2012')
    out_dir = os.path.join(args.base_dir, args.output)
    os.makedirs(out_dir, exist_ok=True)
    metadata = blizzard.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def preprocess_ljspeech(args):
    in_dir = os.path.join(args.base_dir, 'LJSpeech-1.1')
    out_dir = os.path.join(args.base_dir, args.output)
    os.makedirs(out_dir, exist_ok=True)
    metadata = ljspeech.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def preprocess_blizzard2013(args):
    in_dir = os.path.join(args.base_dir, 'database/blizzard2013/segmented')
    out_dir = os.path.join(args.base_dir, args.output)
    os.makedirs(out_dir, exist_ok=True)
    metadata = blizzard2013.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def preprocess_vctk(args):
    in_dir = os.path.join(args.base_dir, 'VCTK-Corpus')
    out_dir = os.path.join(args.base_dir, args.output)
    os.makedirs(out_dir, exist_ok=True)
    metadata = vctk.build_from_path(hparams, in_dir, out_dir, args.num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def preprocess_zeroth(args):
    in_dir = os.path.join(args.base_dir, 'zeroth_korean')
    out_dir = os.path.join(args.base_dir, args.output)
    os.makedirs(out_dir, exist_ok=True)
    metadata = zeroth.build_from_path(hparams, in_dir, out_dir, args.num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def preprocess_seoul(args):
    in_dir = os.path.join(args.base_dir, 'seoul/clean')
    out_dir = os.path.join(args.base_dir, args.output)
    os.makedirs(out_dir, exist_ok=True)
    metadata = seoul.build_from_path(hparams, in_dir, out_dir, args.num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            if m is not None:
                f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([m[2] for m in metadata if m is not None])
    hours = frames * hparams.frame_shift_ms / (3600 * 1000)
    print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frames, hours))
    print('Max input length:  %d' % max(len(m[3]) for m in metadata))
    print('Max output length: %d' % max(m[2] for m in metadata))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.getcwd())
    parser.add_argument('--output', default='training')
    parser.add_argument('--dataset', required=True, choices=['blizzard', 'ljspeech', 'blizzard2013', 'vctk', 'zeroth', 'seoul'])
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    args = parser.parse_args()
    if args.dataset == 'blizzard':
        preprocess_blizzard(args)
    elif args.dataset == 'ljspeech':
        preprocess_ljspeech(args)
    elif args.dataset == 'blizzard2013':
        preprocess_blizzard2013(args)
    elif args.dataset == 'vctk':
        preprocess_vctk(args)
    elif args.dataset == 'zeroth':
        preprocess_zeroth(args)
    elif args.dataset == 'seoul':
        preprocess_seoul(args)


if __name__ == "__main__":
    main()
