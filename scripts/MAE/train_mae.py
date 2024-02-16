import numpy as np
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

import models_mae

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# データセットのパス
dataset_path = "/takaya_workspace/Medical_AI/data/RadImageNet"

# データセットの読み込み
dataset = RadImageNet(root=dataset_path)