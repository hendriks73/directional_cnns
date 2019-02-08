#!/bin/bash

#
# Helper script to train and predict on Google ML Engine.
#
# To make this work, upload your data to Google Cloud Storage.
# Note that *.joblib files need to be created from your audio files.
#
# Then adjust the variables below to match your region and data layout.
# Finally, run the script.
#
# Predictions will be stored in the remote folder JOB_DIR.
#


export BUCKET_NAME=directional_cnns
export REGION=europe-west1
export JOB_NAME="directional_key_cnn_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export MODEL_DIR=gs://$BUCKET_NAME/key_models/

export TRAIN_FILE=gs://$BUCKET_NAME/key_train.tsv
export VALID_FILE=gs://$BUCKET_NAME/key_valid.tsv

export FEATURE_FILES=gs://$BUCKET_NAME/giantsteps_key.joblib,gs://$BUCKET_NAME/mtg_tempo_key.joblib,gs://$BUCKET_NAME/gtzan_key.joblib,gs://$BUCKET_NAME/lmd_key.joblib
export TEST_FILES=gs://$BUCKET_NAME/giantsteps-key.tsv,gs://$BUCKET_NAME/gtzan_key.tsv,gs://$BUCKET_NAME/lmd_key_test.tsv

gcloud ml-engine jobs submit training $JOB_NAME --module-name=tempocnntalk.training --region=$REGION --package-path=./tempocnntalk --job-dir=$JOB_DIR --runtime-version=1.10 --config=./cloudml-gpu.yaml -- --train-file=$TRAIN_FILE --valid-file=$VALID_FILE --test-files=$TEST_FILES --feature-files=$FEATURE_FILES --model-dir=$MODEL_DIR
