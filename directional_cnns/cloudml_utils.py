"""
Utility functions for dealing with Google ML Engine (pseudo-) transparently.
"""

import re
from os import makedirs, remove
from os.path import dirname, isdir
from random import randint

from sklearn.externals.joblib import dump
from tensorflow.python.lib.io.file_io import file_exists, FileIO


def create_local_copy(remote_file):
    """
    Create a local copy for the given, potentially remote file.
    Does nothing, if the file is already local or a local file
    with the same name already exists.

    :param remote_file: potentially remote file
    :return: local file
    """
    local_file = remote_file
    if not file_exists(remote_file):
        print('File does not exist: {}'.format(remote_file))

    if is_remote(remote_file):
        local_file = to_local(remote_file)
        if file_exists(local_file):
            print('Local file already exists: {}'.format(local_file))
        else:
            print('Writing to local file: {}'.format(local_file))
            copy(remote_file, local_file)
    else:
        print('Local copy for {} not needed.'.format(remote_file))
    return local_file


def copy(source, dest):
    """
    Copy from source to dest, create all necessary dirs.

    :param source: source file
    :param dest: dest file
    """
    with FileIO(source, mode='rb') as input_f:
        make_missing_dirs(dest)
        with open(dest, mode='wb') as output_f:
            while 1:
                buf = input_f.read(1024 * 1024)
                if not buf:
                    break
                output_f.write(buf)


def is_remote(file):
    """
    Does the filename start with 'gs://'?

    :param file:
    :return: true or false
    """

    return file is not None and file.startswith('gs://')


def to_local(file):
    """
    Remove the ``gs://BUCKET/`` prefix.

    :param file: file
    :return: relative file name
    """
    pattern = re.compile('gs://[^/]+/(.*)')
    match = pattern.match(file)
    return match.group(1)


def save_model(model, file):
    """
    Save model to the given file (potentially Google storage).

    :param model: model
    :param file: output file
    """
    print('Saving model to file {}.'.format(file))
    # do not overwrite
    if file_exists(file):
        raise FileExistsError('File {} already exists.'.format(file))

    temp_file = 'temp_model_{}.h5'.format(randint(0, 100000000))
    model.save(temp_file)
    try:
        copy_to_remote(temp_file, file)
    finally:
        remove(temp_file)


def dump_joblib(data, file):
    """
    Save data to the given file (potentially Google storage).

    :param data: data
    :param file: output file
    """
    print('Saving data to file {}.'.format(file))
    # do not overwrite
    if file_exists(file):
        raise FileExistsError('File {} already exists.'.format(file))

    temp_file = 'temp_joblib_{}.joblib'.format(randint(0, 100000000))
    dump(data, temp_file)
    try:
        copy_to_remote(temp_file, file)
    finally:
        remove(temp_file)


def copy_to_remote(source, destination):
    """
    Copy data to google storage

    :param source: source
    :param destination: dest
    """
    with FileIO(source, mode='rb') as input_f:
        with FileIO(destination, mode='wb') as output_f:
            output_f.write(input_f.read())


def make_missing_dirs(file):
    """
    Create missing directories.

    :param file: file
    """
    if '/' in file and not isdir(dirname(file)):
        makedirs(dirname(file))