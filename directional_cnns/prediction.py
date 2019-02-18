"""
Predict labels with a model.
"""
import numpy as np


def predict(model, input_shape, windowed, ground_truth, features, normalizer):
    """
    Predict values using the given model, spectrogram input shape, ground truth,
    features dict and normalizer function.

    :param model: model
    :param input_shape: single spectrogram shape
    :param windowed: if ``true``, predict for multiple windows
    :param ground_truth: ground truth necessary to convert indices to actual labels
    :param features: dict, mapping keys to spectrograms
    :param normalizer: function that takes a numpy array and normalizes it (needed before prediction)
    :return: dict, mapping keys to predicted labels
    """
    results = {}
    for key in ground_truth.labels.keys():
        # make sure we don't modify the original!
        spectrogram = np.copy(features[key])

        effective_shape = input_shape
        # cropping for key
        if input_shape[0] != spectrogram.shape[1] and input_shape[0] == 168:
            spectrogram = spectrogram[8:-16]
            effective_shape = (spectrogram.shape[0], input_shape[1], input_shape[2])

        if windowed:
            length = spectrogram.shape[1]
            nb_samples = length // input_shape[1]
            samples = []
            for i in range(nb_samples):
                sample = spectrogram[:, input_shape[1] * i:input_shape[1] * (i+1),:]
                if normalizer is not None:
                    sample = normalizer(sample)
                samples.append(np.reshape(sample, (1, *effective_shape)))
            X = np.vstack(samples)
        else:
            # this assumes that we can predict spectrograms of arbitrary lengths (dim=1)
            if normalizer is not None:
                spectrogram = normalizer(spectrogram)
            X = np.expand_dims(spectrogram, axis=0)

        predictions = model.predict(X, X.shape[0])
        predictions = np.sum(predictions, axis=0)
        index = np.argmax(predictions)
        results[key] = ground_truth.get_label(index)
    return results
