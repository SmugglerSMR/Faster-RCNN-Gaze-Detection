#!/bin/bash -l

#PBS -N EYE_FullView_10
#PBS -l ncpus=8
#PBS -l mem=16GB
#PBS -l walltime=10:00:00

module load tensorflow/1.1.0-intel-2017a-python-3.6.1

cd /home/n9312706/egh455/FRCNN-Inception

export PYTHONPATH=$(pwd):$(pwd)/slim

pip install absl-py --user
pip install pillow --user

python object_detection/legacy/train.py --train_dir=train_incep_10 --pipeline_config_path=faster_rcnn_inception_v2_coco_eye.config