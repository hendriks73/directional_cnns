"""
Functions for loading and augmenting samples.
"""

import math

import numpy as np
from tensorflow.python.keras.preprocessing.image import apply_transform


def create_mel_sample_loader(dataset, shape=(40, 256, 1), random_offset=True, normalizer=None):
    """
    Create sample loading function that is capable of loading and augmenting a sample
    based on an augmentation parameter. The loader function is suitable for the Mel spectrograms
    we use for tempo estimation.

    :param normalizer: normalization function
    :param shape: desired shape
    :param random_offset: return the first X frames or randomly chosen X frames, X=shape[1]
    :param dataset: dataset, dict that maps a ``key`` to a sample
    :return: function that returns a sample for a key (or sample id)
    """
    def mel_sample_loader(key, augmentation_parameter=1.):
        scale_factor = augmentation_parameter
        if scale_factor is None:
            scale_factor = 1.
        mel_spectrogram = dataset[key]
        # we want (40, length)
        offset = 0
        length = shape[1]
        max_length_after_scaling = max(length, math.ceil(length * scale_factor))
        if mel_spectrogram.shape[1] < max_length_after_scaling:
            # the source is too short, let's pad with zeros
            # better with np.pad()?
            copy = np.zeros((mel_spectrogram.shape[0], max_length_after_scaling, mel_spectrogram.shape[2]))
            copy[:, :mel_spectrogram.shape[1], :] = mel_spectrogram
            mel_spectrogram = copy
        if random_offset and max_length_after_scaling < mel_spectrogram.shape[1]:
            # only compute random offset, if we really have material to work with
            offset = np.random.randint(0, mel_spectrogram.shape[1] - max_length_after_scaling)
        if scale_factor is not None and scale_factor != 1.:
            unscaled = mel_spectrogram[:, offset:offset + max_length_after_scaling]
            data = scale(unscaled, scale_factor)
        else:
            data = mel_spectrogram[:, offset:offset + length]
        if normalizer is not None:
            data = normalizer(data)

        return data

    def scale(mel_spectrogram, scale_factor):
        w = mel_spectrogram.shape[0]
        h = mel_spectrogram.shape[1]
        transform = np.array([[scale_factor, 0, 0], [0, 1, 0], [0, 0, 1]])
        image = mel_spectrogram.astype(np.float64).reshape(w, h)
        scaled_image = apply_transform(image, transform, fill_mode='constant').astype(np.float16)
        # if the result is shorter, shorten the np array as well
        if scale_factor > 1:
            new_length = int(scaled_image.shape[1] / scale_factor)
            scaled_image = scaled_image[:, :new_length, ]
        mel_spectrogram = np.expand_dims(scaled_image.astype(mel_spectrogram.dtype), axis=2)
        return mel_spectrogram

    return mel_sample_loader


def create_cq_sample_loader(dataset, shape=(168, 60, 1), random_offset=True, normalizer=None):
    """
    Create sample loading function that is capable of loading and augmenting a sample
    based on an augmentation parameter. The loader function is suitable for the constant Q spectrograms
    we use for key estimation.

    :param normalizer: normalization function
    :param shape: desired shape
    :param random_offset: return the first X frames or randomly chosen X frames, X=shape[1]
    :param dataset: dataset, dict that maps a ``key`` to a sample
    :return: function that returns a sample for a key (or sample id)
    """
    def cq_sample_loader(key, augmentation_parameter=0):
        if augmentation_parameter is None:
            bin_shift = 0
        else:
            bin_shift = augmentation_parameter * 2  # , because we have two bins per semitone
        cq_spectrogram = dataset[key]
        offset = 0
        length = shape[1]
        if random_offset and cq_spectrogram.shape[1] > length:
            offset = np.random.randint(0, cq_spectrogram.shape[1] - length)
        if bin_shift is not None:
            # possible shifts: -4 ... +7 -> -8 ... +14 (two bins per semitone)
            data = cq_spectrogram[8+bin_shift:-(16-bin_shift), offset:offset + length]
        else:
            data = cq_spectrogram[8:-16, offset:offset + length]
        if normalizer is not None:
            data = normalizer(data)

        return data

    return cq_sample_loader