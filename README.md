[![CC BY 3.0](https:/img.shields.io/badge/License-CC%20BY%203.0-blue.svg)](https:/creativecommons.org/licenses/by/3.0/)

# Directional CNNs

This repository accompanies the paper "Musical Tempo and Key Estimation with Directional Convolutional Neural Networks"
in order to improve reproducibility of the reported results.

## Audio Files

Unfortunately, because of size limitations imposed by GitHub as well as copyright issues, this repository does not
contain all audio samples or extracted features. But you can download those and extract them yourself.

Download links: 

1. GTzan: http:/marsyas.info/download/data_sets/
2. Ballroom: http:/mtg.upf.edu/ismir2004/contest/tempoContest/node5.html
3. Extended Ballroom: http:/anasynth.ircam.fr/home/media/ExtendedBallroom
4. GiantSteps Key: https:/github.com/GiantSteps/giantsteps-key-dataset
5. GiantSteps Tempo: https:/github.com/GiantSteps/giantsteps-tempo-dataset
6. GiantSteps MTG Key and Tempo: https:/github.com/GiantSteps/giantsteps-mtg-key-dataset
7. LMD Key and Tempo: https:/bit.ly/2Bl8D1J

Should you use any of the datasets in your academic work, please cite the corresponding publications.  

## Annotations

All necessary annotations are in the [annotations](./annotations) folder. For easy parsing they are formatted
in a simple tab separated values (`.tsv`) format, with columns `id \t bpm \t key \t genre`. The class
[GroundTruth](./directional_cnns/groundtruth.py) is capable of reading and interpreting these files.  

## Installation

In a clean Python 3.5/3.6 environment:

    git clone https:/github.com/XXX/directional_cnns.git
    cd directional_cnns
    python setup.py install

## Feature Extraction

To extract features, please use the code in [feature_extraction.py](./directional_cnns/feature_extraction.py).
Depending on how you define sample identifiers, you may need to make some manual adjustments.
The created `.joblib` files are simple dictionaries, containing strings as keys and a spectrograms as values.
Note that the extracted spectrograms for the key and the tempo task differ (CQT vs Mel).

After installation, you may run the extraction using the following command line script:

    directional_cnn_extraction -a AUDIO_FILES_FOLDER [-g GROUND_TRUTH.tsv]
    
The ground truth file is optional. If given, only files that also occur in the ground truth are added
to the created feature `.joblib` files.

## Running

You can run the code either locally or on [Google ML Engine](https:/gcpsignup.page.link/9kLi).

### Local

Running this locally only makes sense on a GPU and even then it will take very long.  

To run the training/reporting locally, you need to execute the script [training.py](./directional_cnns/training.py)
with the following parameters (example for *key*):

    --job-dir=./
    --model-dir=./
    --train-file=annotations/key_train.tsv --valid-file=annotations/key_valid.tsv
    --test-files=annotations/giantsteps-key.tsv,annotations/gtzan_key.tsv,annotations/lmd_key_test.tsv
    --feature-files=features/giantsteps_key.joblib,features/mtg_tempo_key.joblib,features/gtzan_key.joblib,features/lmd_key.joblib

After installation, you may run the training code using the following command line script:

    directional_cnn_training [arguments]


### Remote

To run the training/reporting remotely on [Google ML Engine](https:/gcpsignup.page.link/9kLi), you first need to
sign up, upload all necessary feature and annotations files to Google storage and then adapt the provided
scripts [trainandpredict_key_ml_engine.sh](./trainandpredict_key_ml_engine.sh) and
[trainandpredict_tempo_ml_engine.sh](./trainandpredict_tempo_ml_engine.sh) accordingly.

## License

This repository is licensed under [CC BY 3.0](https:/creativecommons.org/licenses/by/3.0/).
For attribution, please cite:

> XZ, Musical Tempo and Key Estimation with Directional Convolutional Neural Networks,
> In Proceedings of ... 