import os
import gc
import random
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image
from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedKFold

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F

import timm

from torchvision import transforms as T
from radimagenet_dataset import RadImageNet, RadImageNetForSupCon
from models_supcon import SupConModel
from utils import TwoCropTransform, save_image
from loss import HMLC

transforms = T.Compose([
    T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=15),
    T.Resize((224, 224)),
    T.ToTensor(),
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# データセットのパス
dataset_path = "/takaya_workspace/Medical_AI/data/RadImageNet"

# データセットの読み込み
dataset = RadImageNetForSupCon(root=dataset_path, transform=TwoCropTransform(transforms))
train_size = int(len(dataset) * 0.9)
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

bsz = 64
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, shuffle=True)

BASE_MODEL = 'resnet18d'
model = SupConModel(base_name=BASE_MODEL, pretrained=False)
model.to(device)

criterion =HMLC()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

epochs = 2

for epoch in range(epochs):
    model.train()
    for i, (images, labels, img_paths) in enumerate(train_dataloader):
        images = torch.cat([images[0], images[1]], dim=0)
        images = images.to(device)
        labels = labels.to(device)
        #print(images.shape, labels.shape)
        optimizer.zero_grad()
        
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        #print(features.shape, labels.shape)
        # ラベルは１次元だけ取り出す
        labels = labels
        loss = criterion(features, labels)
        print(loss)
        loss.backward()
        optimizer.step()