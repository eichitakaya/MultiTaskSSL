#!/bin/bash

# --model_pathはシェルスクリプトの引数として渡し、--datasetを変更して順に実行する。
model_path="/takaya_workspace/MultiTaskSSL/result/RadImageNet_Classification165_ViT/model.pth"
epochs=10
batchsize=32

python downstream_evaluation.py --dataset=thyroid --model_path=$model_path --epochs=$epochs --batchsize=$batchsize
python downstream_evaluation.py --dataset=breast --model_path=$model_path --epochs=$epochs --batchsize=$batchsize
python downstream_evaluation.py --dataset=acl --model_path=$model_path --epochs=$epochs --batchsize=$batchsize