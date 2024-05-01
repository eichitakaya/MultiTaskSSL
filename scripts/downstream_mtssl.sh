#!/bin/bash

# --model_pathはシェルスクリプトの引数として渡し、--datasetを変更して順に実行する。
model_path="/takaya_workspace/MultiTaskSSL/result/MTSSL_ViT/model_epoch4.pth"
epochs=10
batchsize=64

python downstream_evaluation_mtssl.py --dataset=thyroid --model_path=$model_path --epochs=$epochs --batchsize=$batchsize
python downstream_evaluation_mtssl.py --dataset=breast --model_path=$model_path --epochs=$epochs --batchsize=$batchsize
python downstream_evaluation_mtssl.py --dataset=acl --model_path=$model_path --epochs=$epochs --batchsize=$batchsize