"""
RadImageNetにおける165値の分類問題を解く
"""
# ライブラリのインポート
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt

from dataloader import RadImageNet

# データセットのパス
dataset_path = "/takaya_workspace/Medical_AI/data/RadImageNet"

# データセットの読み込み
dataset = RadImageNet(root=dataset_path)

# データセットの分割
train_size = int(len(dataset) * 0.8)
val_size = int(len(dataset) * 0.1)
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

print("train_dataset: ", len(train_dataset))
print("val_dataset: ", len(val_dataset))
print("test_dataset: ", len(test_dataset))

# データローダーの作成
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

# モデルの定義
model = torchvision.models.resnet18(pretrained=True)
# モデルの最終層の出力ユニットを165に変更
model.fc = nn.Linear(in_features=512, out_features=165, bias=True)

# 損失関数の定義
criterion = nn.CrossEntropyLoss()

# 最適化手法の定義
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 学習率のスケジューラーの定義
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# デバイスの指定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# モデルをGPUに転送
model.to(device)

# 学習
num_epochs = 10
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch+1, num_epochs))
    print("----------")
    # train
    model.train()
    train_loss = 0.0
    train_corrects = 0
    for inputs, labels, paths in train_dataloader:
        inputs = inputs.to(device)
        labels = labels[2].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        train_corrects += torch.sum(preds == labels.data)
    train_loss = train_loss / len(train_dataset)
    train_acc = train_corrects.double() / len(train_dataset)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    print("train loss: {:.4f}, train acc: {:.4f}".format(train_loss, train_acc))
    # val
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    for inputs, labels, paths in val_dataloader:
        inputs = inputs.to(device)
        labels = labels[2].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        val_loss += loss.item() * inputs.size(0)
        val_corrects += torch.sum(preds == labels.data)
    val_loss = val_loss / len(val_dataset)
    val_acc = val_corrects.double() / len(val_dataset)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
    print("val loss: {:.4f}, val acc: {:.4f}".format(val_loss, val_acc))
    scheduler.step()