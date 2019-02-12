"""
Encapsulates both ground truth and evaluation accuracy reports.

(needs refactoring - didn't have time :-( )
"""

import csv
import gc
import math
from os.path import basename, splitext

import librosa
import numpy as np
from sklearn.externals.joblib import load
from tensorflow.python.lib.io.file_io import file_exists
from tensorflow.python.keras import backend as K

from directional_cnns.cloudml_utils import create_local_copy, dump_joblib


class GroundTruth:

    def consistent_model_errors(self, model_errors):
        consistent_model_errors = model_errors[0]
        for e in model_errors[1:]:
            for k, v in e.items():
                if consistent_model_errors[k] != v:
                    consistent_model_errors[k] = -1
        return consistent_model_errors

    def create_error_reports(self, correct_errors, musical_errors, errors, sorted_names):
        plain_report = ''
        latex_report = ''
        # what kind of error report
        consistent_errors = {name: self.consistent_model_errors(error) for name, error in errors.items()}
        for name in sorted_names:
            a = float(len(consistent_errors[name]))
            c = sum(value != -1 for value in consistent_errors[name].values())
            plain_report += 'consistent errors for {}:\n{:.3%} {}\n'.format(name, c / a, consistent_errors[name])
        plain_report += '---------------------------------------------------------------------------------\n'
        for a in range(len(sorted_names)):
            model_name_1 = sorted_names[a]
            consistent_errors_1 = consistent_errors[model_name_1]
            for b in range(a+1, len(sorted_names)):
                model_name_2 = sorted_names[b]
                consistent_errors_2 = consistent_errors[model_name_2]
                correct = []
                same_musical_error = []
                same_error = []
                both_inconsistent = []
                different_error = []
                correct_incorrect = []
                incorrect_correct = []

                for k in consistent_errors_1.keys():
                    err1 = consistent_errors_1[k]
                    err2 = consistent_errors_2[k]
                    if err1 in correct_errors and err2 in correct_errors:
                        correct.append(k)
                    if err1 == err2 and err1 == -1:
                        both_inconsistent.append(k)
                    if err1 == err2 and err1 in musical_errors:
                        same_musical_error.append(k)
                    if err1 == err2 and err1 not in correct_errors and err1 >= 0:
                        same_error.append(k)
                    if err1 != err2 and err1 not in correct_errors and err2 not in correct_errors:
                        different_error.append(k)
                    if err1 in correct_errors and err2 not in correct_errors:
                        correct_incorrect.append(k)
                    if err2 in correct_errors and err1 not in correct_errors:
                        incorrect_correct.append(k)
                plain_report += 'comparing {}\nwith      {}:\n'.format(model_name_1, model_name_2)
                all = float(len(consistent_errors_1))
                plain_report += '- both correct {:8.3%}, {}\n'.format(len(correct) / all, correct)
                plain_report += '- both incons  {:8.3%}, {}\n'.format(len(both_inconsistent) / all, both_inconsistent)
                plain_report += '- same error   {:8.3%}, {}\n'.format(len(same_error) / all, same_error)
                plain_report += '- diff error   {:8.3%}, {}\n'.format(len(different_error) / all, different_error)
                plain_report += '- music error  {:8.3%}, {}\n'.format(len(same_musical_error) / all, same_musical_error)
                plain_report += '- 1st correct  {:8.3%}, {}\n'.format(len(correct_incorrect) / all, correct_incorrect)
                plain_report += '- 2nd correct  {:8.3%}, {}\n'.format(len(incorrect_correct) / all, incorrect_correct)
            plain_report += '---------------------------------------------------------------------------------\n'
        return plain_report, latex_report


class TempoGroundTruth(GroundTruth):
    """
    Tempo Ground truth
    """

    def __init__(self, file, nb_classes=256, label_offset=30) -> None:
        super().__init__()
        self.file = file
        self.nb_classes = nb_classes
        self.label_offset = label_offset
        self.labels = self._read_label_file(self.file)
        self.name = file.replace('.tsv', '')

    def classes(self):
        return [i for i in range(self.nb_classes)]

    def get_label(self, index):
        if index < 0 or index > self.nb_classes:
            return None
        return index + self.label_offset

    def get_index(self, label):
        if label < self.label_offset:
            return 0
        if label > self.nb_classes + self.label_offset:
            return self.nb_classes
        return round(label - self.label_offset)

    def get_index_for_key(self, key, scale_factor=1.):
        if scale_factor is None:
            scale_factor = 1.
        label = self.labels[key]
        if label is None:
            return None
        else:
            return self.get_index(label*scale_factor)

    def _read_label_file(self, file):
        labels = {}
        with open(file, mode='r', encoding='utf-8') as text_file:
            reader = csv.reader(text_file, delimiter='\t')
            for row in reader:
                id = row[0]
                bpm = float(row[1])
                labels[id] = bpm
        return labels

    def errors(self, predictions):
        errors = {}
        for key in self.labels.keys():
            if key in predictions:
                predicted_label = predictions[key]
                true_label = self.labels[key]

                acc0 = same_tempo(true_label, predicted_label, tolerance=0.0)
                acc1 = same_tempo(true_label, predicted_label)
                acc2 = acc1 or same_tempo(true_label, predicted_label, factor=2.) \
                       or same_tempo(true_label, predicted_label, factor=1. / 2.) \
                       or same_tempo(true_label, predicted_label, factor=3.) \
                       or same_tempo(true_label, predicted_label, factor=1. / 3.)

                if acc0:
                    error = 0
                elif acc1:
                    error = 1
                elif acc2:
                    error = 2
                else:
                    error = 3

                errors[key] = error
            else:
                print('No prediction for key {}'.format(key))
        return errors

    def accuracy_stats(self, predictions):
        acc1_hist = np.empty(30)
        acc1_hist.fill(0.)
        acc1_hist_true = np.empty(30)
        acc1_hist_true.fill(0.)

        acc0_sum = 0
        acc1_sum = 0
        acc2_sum = 0
        count = 0
        for key in self.labels.keys():
            if key in predictions:
                predicted_label = predictions[key]
                true_label = self.labels[key]

                acc0 = same_tempo(true_label, predicted_label, tolerance=0.0)
                acc1 = same_tempo(true_label, predicted_label)
                acc2 = acc1 or same_tempo(true_label, predicted_label, factor=2.) \
                       or same_tempo(true_label, predicted_label, factor=1. / 2.) \
                       or same_tempo(true_label, predicted_label, factor=3.) \
                       or same_tempo(true_label, predicted_label, factor=1. / 3.)

                if acc0:
                    acc0_sum += 1
                if acc1:
                    acc1_sum += 1
                if acc2:
                    acc2_sum += 1

                acc1_hist_true[int(true_label / 10)] += 1.
                if acc1:
                    acc1_hist[int(true_label / 10)] += 1.
            else:
                print('No prediction for key {}'.format(key))

            count += 1
        acc0_result = acc0_sum / float(count)
        acc1_result = acc1_sum / float(count)
        acc2_result = acc2_sum / float(count)
        acc1_hist_result = acc1_hist / acc1_hist_true
        # combine into one array
        result = [acc0_result, acc1_result, acc2_result]
        for i in range(30):
            if math.isnan(acc1_hist_result[i]):
                result.append(-1.)
            else:
                result.append(acc1_hist_result[i])
        return result

    def create_accuracy_reports(self, features, input_shape, windowed, log, models, normalizer, test_files, predictor):
        plain_report = 'windowed={}\n'.format(windowed) \
                       + 'Testset | Runs | mean Acc0 | mean Acc1 | mean Acc2 | model\n' \
                       + '---------------------------------------------------------------------------------\n'
        latex_report = '% windowed={}\n'.format(windowed)\
                      + 'Testset & Runs & mean Acc0 & mean Acc1 & mean Acc2 & model \\\\\n \\hline \\\\\n'
        plain_table_template = '{:<16} | {} | {:8.3%} ({:.5f}) | {:8.3%} ({:.5f}) | {:8.3%} ({:.5f}) | {}\n'
        latex_table_template = plain_table_template.replace('|', '&').replace('\n', '  \\\\\n')

        latex_data_table = '% windowed={}\n'.format(windowed)\
                           + '\pgfplotstableread[row sep=\\\\,col sep=&]{\n'\
                           + 'Testset & Runs & Parameters & Acc1 & Std1 & Model \\\\\n'
        latex_data_table_template = '{:<16} & {} & {:10d} & {:.5f} & {:.5f} & {} \\\\\n'

        # consistently correct - not interesting
        # consistently misclassified - interesting: wrong in groundtruth?
        # inconsistently misclassified - interesting: why better in one alg than another?
        # correctly classified, though bad design: why still successful (better than chance!?)
        sorted_names = list(models.keys())
        sorted_names.sort()

        for test_file in test_files:
            errors = {}
            log('----------')
            log(test_file)
            test_ground_truth = TempoGroundTruth(test_file)
            for model_name in sorted_names:
                log(model_name)
                same_kind_models = models[model_name]
                same_kind_errors = []
                errors[model_name] = same_kind_errors
                accuracies = []
                param_count = 0
                for run, model_loader in enumerate(same_kind_models):
                    log('Loading model {} from disk...'.format(model_loader.file))
                    model = model_loader.load()
                    param_count = model.count_params()

                    test_name = splitext(basename(test_file))[0]
                    predictions_file = model_loader.file.replace('.h5', '_pred_{}.joblib'.format(test_name))
                    if file_exists(predictions_file):
                        log('Predictions file {} already exists. Loading predictions.'.format(predictions_file))
                        predictions = load(create_local_copy(predictions_file))
                    else:
                        predictions = predictor(model, input_shape, windowed, test_ground_truth, features, normalizer)
                        dump_joblib(predictions, predictions_file)

                    log('{}. run {}:\n{}'.format(run, model_name, predictions))
                    same_kind_errors.append(test_ground_truth.errors(predictions))
                    acc = test_ground_truth.accuracy_stats(predictions)
                    log(str(acc))
                    accuracies.append(np.array(acc))
                    # don't keep all models in memory
                    del model
                    # don't keep predictions
                    del predictions
                    K.clear_session()
                    gc.collect()
                np_acc = np.vstack(accuracies)
                means = np.mean(np_acc, axis=0)
                stdevs = np.std(np_acc, axis=0)
                log('means  : ' + str(means.tolist()))
                log('stddevs: ' + str(stdevs.tolist()))

                latex_report += latex_table_template.format(test_ground_truth.name,
                                                      len(same_kind_models),
                                                      means[0], stdevs[0],
                                                      means[1], stdevs[1],
                                                      means[2], stdevs[2],
                                                      model_name)

                latex_data_table += latex_data_table_template.format(test_ground_truth.name,
                                                      len(same_kind_models),
                                                      param_count,
                                                      means[1], stdevs[1],
                                                      model_name)

                plain_report += plain_table_template.format(test_ground_truth.name,
                                                      len(same_kind_models),
                                                      means[0], stdevs[0],
                                                      means[1], stdevs[1],
                                                      means[2], stdevs[2],
                                                      model_name)
            plain_report += '---------------------------------------------------------------------------------\n'

            plain, latex = self.create_error_reports({0, 1}, {2}, errors, sorted_names)
            plain_report += plain
            latex_report += latex

        latex_data_table += '}\\tempoaccuracy\n'
        latex_report += '\n\n' + latex_data_table + '\n\n'
        return latex_report, plain_report


class KeyGroundTruth(GroundTruth):
    """
    Key Ground truth
    """

    def __init__(self, file, nb_classes=24) -> None:
        super().__init__()
        self.file = file
        self.nb_classes = nb_classes
        self.labels = self._read_label_file(self.file)
        self.name = file.replace('.tsv', '')

    def classes(self):
        return [i for i in range(self.nb_classes)]

    def get_label(self, index):
        if index < 0 or index > self.nb_classes:
            return None
        minor = index >= 12
        midi = index + 12
        if minor:
            midi = index - 12
        label = librosa.midi_to_note(midi=midi, octave=False)
        if minor:
            label += 'm'
        return label

    def get_index(self, label):
        if label is None:
            return None
        try:
            klass = librosa.note_to_midi(label.replace('m', '')) - 12
            if label.endswith('m'):
                klass += 12
            return klass
        except librosa.ParameterError:
            return None

    def get_index_for_key(self, key, semitone_shift=0):
        if semitone_shift is None:
            semitone_shift = 0
        label = self.labels[key]
        if label is None:
            return None
        else:
            index = self.get_index(label)
            if index >= 12:
                index = self.shift(index-12, -semitone_shift) + 12
            else:
                index = self.shift(index, -semitone_shift)
            return index

    @staticmethod
    def shift(index, semitone_shift):
        return (index + semitone_shift + 12) % 12

    def _read_label_file(self, file):
        labels = {}
        with open(file, mode='r', encoding='utf-8') as text_file:
            reader = csv.reader(text_file, delimiter='\t')
            for row in reader:
                id = row[0]
                key = row[2]
                labels[id] = key
        return labels

    def errors(self, predictions):
        errors = {}
        for key in self.labels.keys():
            if key in predictions:
                predicted_label = predictions[key]
                true_label = self.labels[key]

                correct = same_key(true_label, predicted_label)
                fifth = same_key(true_label, predicted_label, semitone_distance=7) or same_key(true_label, predicted_label, semitone_distance=-7)
                relative = same_key(true_label, predicted_label, semitone_distance=-3, same_mode=False, true_major=True)\
                           or same_key(true_label, predicted_label, semitone_distance=3, same_mode=False, true_major=False)
                parallel = same_key(true_label, predicted_label, semitone_distance=0, same_mode=False)

                if correct:
                    error = 0
                elif fifth:
                    error = 1
                elif relative:
                    error = 2
                elif parallel:
                    error = 3
                elif parallel:
                    error = 4
                else:
                    error = 5

                errors[key] = error
            else:
                print('No prediction for key {}'.format(key))
        return errors

    def accuracy_stats(self, predictions):
        correct_sum = 0
        fifth_sum = 0
        relative_sum = 0
        parallel_sum = 0
        count = 0
        for key in self.labels.keys():
            if key in predictions:
                predicted_label = predictions[key]
                true_label = self.labels[key]
                correct = same_key(true_label, predicted_label)
                fifth = same_key(true_label, predicted_label, semitone_distance=7) or same_key(true_label, predicted_label, semitone_distance=-7)
                relative = same_key(true_label, predicted_label, semitone_distance=-3, same_mode=False, true_major=True)\
                           or same_key(true_label, predicted_label, semitone_distance=3, same_mode=False, true_major=False)
                parallel = same_key(true_label, predicted_label, semitone_distance=0, same_mode=False)

                if correct:
                    correct_sum += 1
                if fifth:
                    fifth_sum += 1
                if relative:
                    relative_sum += 1
                if parallel:
                    parallel_sum += 1
            else:
                print('No prediction for key {}'.format(key))

            count += 1
        correct_result = correct_sum / float(count)
        fifth_result = fifth_sum / float(count)
        relative_result = relative_sum / float(count)
        parallel_result = parallel_sum / float(count)
        score_result = (correct_sum + 0.5*fifth_sum + 0.3*relative_sum + 0.2*parallel_sum) / float(count)
        return score_result, correct_result, fifth_result, relative_result, parallel_result

    def create_accuracy_reports(self, features, input_shape, windowed, log, models, normalizer, test_files, predictor):
        plain_report = 'windowed={}\n'.format(windowed) \
                       + 'Testset | Runs | mean Scor | mean Corr | mean Fift | mean Rela | mean Para | model\n' \
                       + '---------------------------------------------------------------------------------\n'
        latex_report = '% windowed={}\n'.format(windowed)\
                       + 'Testset & Runs & mean Scor & mean Corr & mean Fift & mean Rela & mean Para & model \\\\\n \\hline \\\\\n'

        plain_table_template = '{:<16} | {} | {:8.3%} ({:.5f}) | {:8.3%} ({:.5f}) | {:8.3%} ({:.5f}) | {:8.3%} ({:.5f}) | {:8.3%} ({:.5f}) | {}\n'
        latex_table_template = plain_table_template.replace('|', '&').replace('\n', '  \\\\\n')

        latex_data_table = '% windowed={}\n'.format(windowed)\
                           + '\pgfplotstableread[row sep=\\\\,col sep=&]{\n'\
                           + 'Testset & Runs & Parameters & Acc & AccStd & Score & ScoreStd & Model \\\\\n'
        latex_data_table_template = '{:<16} & {} & {:10d} & {:.5f} & {:.5f} & {:.5f} & {:.5f} & {} \\\\\n'

        sorted_names = list(models.keys())
        sorted_names.sort()

        for test_file in test_files:
            errors = {}
            log('----------')
            log(test_file)
            test_ground_truth = KeyGroundTruth(test_file)
            for model_name in sorted_names:
                log(model_name)
                same_kind_models = models[model_name]
                same_kind_errors = []
                errors[model_name] = same_kind_errors
                accuracies = []
                param_count = 0

                for run, model_loader in enumerate(same_kind_models):
                    log('Loading model {} from disk...'.format(model_loader.file))
                    model = model_loader.load()
                    param_count = model.count_params()

                    test_name = splitext(basename(test_file))[0]
                    predictions_file = model_loader.file.replace('.h5', '_pred_{}.joblib'.format(test_name))
                    if file_exists(predictions_file):
                        log('Predictions file {} already exists. Loading predictions.'.format(predictions_file))
                        predictions = load(create_local_copy(predictions_file))
                    else:
                        predictions = predictor(model, input_shape, windowed, test_ground_truth, features, normalizer)
                        dump_joblib(predictions, predictions_file)

                    same_kind_errors.append(test_ground_truth.errors(predictions))
                    log('{}. run {}:\n{}'.format(run, model_name, predictions))
                    acc = test_ground_truth.accuracy_stats(predictions)
                    log(str(acc))
                    accuracies.append(np.array(acc))
                    # don't keep all models in memory
                    del model
                    # don't keep predictions
                    del predictions
                    K.clear_session()
                    gc.collect()
                np_acc = np.vstack(accuracies)
                means = np.mean(np_acc, axis=0)
                stdevs = np.std(np_acc, axis=0)
                log('means  : ' + str(means.tolist()))
                log('stddevs: ' + str(stdevs.tolist()))

                latex_report += latex_table_template.format(test_ground_truth.name,
                                                      len(same_kind_models),
                                                      means[0], stdevs[0],
                                                      means[1], stdevs[1],
                                                      means[2], stdevs[2],
                                                      means[3], stdevs[3],
                                                      means[4], stdevs[4],
                                                      model_name)

                latex_data_table += latex_data_table_template.format(test_ground_truth.name,
                                                      len(same_kind_models),
                                                      param_count,
                                                      means[1], stdevs[1],
                                                      means[0], stdevs[0],
                                                      model_name)

                plain_report += plain_table_template.format(test_ground_truth.name,
                                                      len(same_kind_models),
                                                      means[0], stdevs[0],
                                                      means[1], stdevs[1],
                                                      means[2], stdevs[2],
                                                      means[3], stdevs[3],
                                                      means[4], stdevs[4],
                                                      model_name)
            plain_report += '---------------------------------------------------------------------------------\n'

            plain, latex = self.create_error_reports({0}, {1, 2, 3, 4}, errors, sorted_names)
            plain_report += plain
            latex_report += latex

        latex_data_table += '}\\keyaccuracy\n'
        latex_report += '\n\n' + latex_data_table + '\n\n'

        return latex_report, plain_report


def same_tempo(true_value, estimated_value, factor=1., tolerance=0.04):
    if tolerance is None or tolerance == 0.0:
        return round(estimated_value * factor) == round(true_value)
    else:
        return abs(estimated_value * factor - true_value) < true_value * tolerance


def same_key(true_value, estimated_value, semitone_distance=0, same_mode=True, true_major=None):
    # convert to ints
    true_minor = true_value.endswith('m')
    true_int = librosa.note_to_midi(true_value.replace('m', ''))
    estimated_minor = estimated_value.endswith('m')
    estimated_int = librosa.note_to_midi(estimated_value.replace('m', ''))

    if true_major is not None:
        if true_major and true_minor:
            return False
        if not true_major and not true_minor:
            return False
    if same_mode and true_minor != estimated_minor:
        return False
    if not same_mode and true_minor == estimated_minor:
        return False
    if estimated_int - true_int != semitone_distance:
        return False
    return True
