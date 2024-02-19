import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.datasets as datasets

from radimagenet_dataset import RadImageNet, RadImageNetForSupCon

import timm

import models_mae

from utils import TwoCropTransform

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
#train_size = int(len(dataset) * 0.9)
#val_size = len(dataset) - train_size
#train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

batchsize = 64
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)
#val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=True)

model = models_mae.__dict__["mae_vit_large_patch16"]()
model.to(device)

criterion =HMLC()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
   
    for i, (images, labels, img_paths) in enumerate(train_dataloader):
        images = torch.cat([images[0], images[1]], dim=0)
        images = images.to(device)
        labels = labels.to(device)
        #print(images.shape, labels.shape)
        optimizer.zero_grad()
        
        loss_mae, features, _, _ = model(images)
        f1, f2 = torch.split(features, [batchsize, batchsize], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        #print(features.shape, labels.shape)
        # ラベルは１次元だけ取り出す
        loss_hmlc = criterion(features, labels)
        loss_multi = loss_mae + loss_hmlc
        print(loss_multi.item(), loss_mae.item(), loss_hmlc.item())
        loss_multi.backward()
        optimizer.step()
    # モデルの保存
    model_path = os.path.join("/takaya_workspace/Medical_AI/MultiTaskSSL/models/mtssl/", "model_epoch" + str(epoch+1) + ".pth")
    torch.save(model.state_dict(), model_path)

        
        