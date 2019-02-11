"""
Feature extraction code.
Creates joblib feature files out of audio files.
"""
import argparse
from os import walk
from os.path import join

import librosa
import numpy as np
import sys
from sklearn.externals import joblib

from directional_cnns.groundtruth import TempoGroundTruth


def extract_tempo_features(file, window_length=1024):
    y, sr = librosa.load(file, sr=11025)
    hop_length = window_length // 2
    data = librosa.feature.melspectrogram(y=y, sr=11025, n_fft=window_length, hop_length=hop_length,
                                          power=1, n_mels=40, fmin=20, fmax=5000)
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    return data.astype(np.float16)


def extract_key_features(file, window_length=8192):
    y, sr = librosa.load(file, sr=22050)
    hop_length = window_length // 2
    octaves = 8
    bins_per_semitone = 2
    bins_per_octave = 12 * bins_per_semitone
    data = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length,
                              fmin=librosa.note_to_hz('C1'),
                              n_bins=bins_per_octave * octaves,
                              bins_per_octave=bins_per_octave))

    data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    return data.astype(np.float16)


def extract_features_from_folder(base_folder, ground_truth, extractor):
    """
    Reads a folder and its subfolders, parses all LOFI.mp3/.wav files and stores
    the result in a dictionary using the filenames (minus the extension) as keys.

    :param base_folder: folder with ``.mp3/.wav`` files
    :return: dictionary with file names as keys
    """
    feature_dataset = {}
    for (dirpath, _, filenames) in walk(base_folder):
        for file in [f for f in filenames if f.endswith('.mp3') or f.endswith('.wav')]:
            key = file.replace('.LOFI.mp3', '').replace('.mp3', '').replace('.wav', '')
            # if we have a ground truth, limit to ids listed in the ground truth
            if ground_truth is not None and key not in ground_truth.labels:
                continue
            features = extractor(join(dirpath, file))
            feature_dataset[key] = features
    return feature_dataset


def convert_audio_folder_to_joblib(base_folder, ground_truth, output_file, extractor):
    """
    Extract features from all audio files in the given folder and its subfolders,
    and store them under keys equivalent to the file names (minus extension),
    store the resulting dict in ``output_file`` and return the dict.

    :param base_folder: base folder for json files
    :param output_file: joblib file
    :return: dict of keys and features
    """
    dataset = extract_features_from_folder(base_folder, ground_truth, extractor)
    joblib.dump(dataset, output_file)
    return dataset


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''This script allows extracting features from mp3 or wav files by recursively
walking a directory tree, starting with a provided base audio folder.
The features are stored in simple dictionaries ('filename_w/o_extension': spectrogram),
which in turn are stored in .joblib files.   

License: CC BY 3.0''')

    parser.add_argument(
        '-a',
        '--audio-folder',
        help='Folder containing mp3 or wav audio files. Will be read recursively, file names are used as keys.',
        required=True
    )
    parser.add_argument(
        '-g',
        '--ground-truth',
        help='TSV file with ground truth. If set, only files also occurring in the truth will be read.',
        required=False
    )
    args = parser.parse_args()
    return args


def main():
    arguments = parse_arguments()

    def tempo_extractor(file):
        return extract_tempo_features(file, window_length=1024)

    def key_extractor(file):
        return extract_key_features(file, window_length=8192)

    audio_folder = arguments.audio_folder
    if arguments.ground_truth is not None:
        ground_truth = TempoGroundTruth(arguments.audio_folder)
    else:
        ground_truth = None

    convert_audio_folder_to_joblib(audio_folder, ground_truth, join(audio_folder, 'tempo_features.joblib'), tempo_extractor)
    convert_audio_folder_to_joblib(audio_folder, ground_truth, join(audio_folder, 'key_features.joblib'), key_extractor)


if __name__ == '__main__':
    main()
