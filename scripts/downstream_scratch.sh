#!/bin/bash

# --model_pathはシェルスクリプトの引数として渡し、--datasetを変更して順に実行する。
epochs=10
batchsize=32

python downstream_evaluation_scratch.py --dataset=thyroid --epochs=$epochs --batchsize=$batchsize
python downstream_evaluation_scratch.py --dataset=breast --epochs=$epochs --batchsize=$batchsize
python downstream_evaluation_scratch.py --dataset=acl --epochs=$epochs --batchsize=$batchsize