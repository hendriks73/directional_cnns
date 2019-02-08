"""
Keras sequence to be used as generator.
"""

import numpy as np
from tensorflow.keras.utils import Sequence


def tempo_augmenter():
    f = [x / 100.0 for x in range(80, 124, 4)]
    return np.random.choice(f)


def key_augmenter():
    return np.random.choice([-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7])


class DataGenerator(Sequence):
    """
    Key-based Keras sequence to be used as generator during fitting and/or predicting.
    This sequence can balance samples for a given set of classes.
    """

    def __init__(self, ground_truth, sample_loader, binarizer, batch_size=32, sample_shape=(32, 32, 32), shuffle=True,
                 augmenter=None):
        """
        Initialization.

        :param scale: scale samples and labels by randomly chosen factors
        :param ground_truth: ground truth containing id (key) and labels
        :param sample_loader: function used to load a sample (features) when given a sample key (e.g. a UUID)
        :param batch_size: batch size
        :param sample_shape: shape of a single feature sample
        :param shuffle: shuffle samples
        """
        self.ground_truth = ground_truth
        self.sample_shape = sample_shape
        self.batch_size = batch_size
        self.sample_ids = list(ground_truth.labels.keys())
        self.binarizer = binarizer
        self.shuffle = shuffle
        self.sample_loader = sample_loader
        self.indexes = None
        self.on_epoch_end()
        self.augmenter = augmenter

    def __len__(self):
        """
        Number of batches per epoch.

        :return: batches per epoch
        """
        return int(np.floor(len(self.sample_ids) / self.batch_size))

    def __getitem__(self, batch_index):
        """
        Generate a batch for the given batch index.

        :param batch_index: batch index
        :return: one batch of data
        """
        indexes = self.indexes[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        temp_keys = [self.sample_ids[k] for k in indexes]
        X, y = self.__data_generation(temp_keys)
        return X, y

    def on_epoch_end(self):
        """
        Re-shuffle, if necessary after each epoch.
        """
        self.indexes = np.arange(len(self.sample_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, keys):
        """
        Creates a batch of data.

        :param keys: ids for the samples that should make up the batch
        :return: batch of data
        """
        X = np.empty((self.batch_size, *self.sample_shape))
        augmentation_parameter = None
        if self.augmenter is not None:
            augmentation_parameter = self.augmenter()

        for i, key in enumerate(keys):
            X[i,] = self.sample_loader(key, augmentation_parameter)

        y = self.binarizer.transform([[self.ground_truth.get_index_for_key(key, augmentation_parameter)] for key in keys])

        return X, y
