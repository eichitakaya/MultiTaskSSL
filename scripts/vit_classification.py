"""
RadImageNetにおける165値の分類問題を解く
モデルはViTを使用
"""
# ライブラリのインポート
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataloader import RadImageNet

batchsize = 512

# データセットのパス
dataset_path = "/takaya_workspace/Medical_AI/data/RadImageNet"

# 結果の保存先（なければmkdir）
result_path = "../result/RadImageNet_Classification165_ViT"
if not os.path.exists(result_path):
    os.mkdir(result_path)

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
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=True)

# ViTモデルの定義
model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
# モデルの最終層の出力ユニットを165に変更
model.fc = nn.Linear(in_features=512, out_features=165, bias=True)

# モデルの並列化
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

# 損失関数の定義
criterion = nn.CrossEntropyLoss()

# 最適化手法の定義
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 学習率のスケジューラーの定義
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# デバイスの指定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# モデルをGPUに転送
model.to(device)

# 学習
num_epochs = 10
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

with tqdm(range(num_epochs)) as pbar_epoch:
    for epoch in pbar_epoch:
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("----------")
        # train
        model.train()
        train_loss = 0.0
        train_corrects = 0
        pbar_epoch.set_description("[Epoch %d]" % (epoch+1))
        with tqdm(enumerate(train_dataloader), total=len(train_dataloader)) as pbar_batch:
            for i, (inputs, labels, paths) in pbar_batch:
                inputs = inputs.to(device)
                labels = labels[:, 2].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                train_corrects += torch.sum(preds == labels.data)
                #print(loss.item())
                
        train_loss = train_loss / len(train_dataset)
        train_acc = train_corrects.double() / len(train_dataset)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        print("train loss: {:.4f}, train acc: {:.4f}".format(train_loss, train_acc))
        
        # val
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        #with tqdm(enumerate(val_dataloader), total=len(val_dataloader)) as pbar_batch:
        for inputs, labels, paths in val_dataloader:
            inputs = inputs.to(device)
            labels = labels[:, 2].to(device)
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

# 学習曲線の描画
plt.figure()
plt.plot(range(num_epochs), train_loss_list, color="blue", linestyle="-", label="train_loss")
plt.plot(range(num_epochs), val_loss_list, color="green", linestyle="--", label="val_loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Learning Curve")
plt.savefig(result_path + "/loss.png")

# モデルの保存
torch.save(model.state_dict(), result_path + "/model.pth")

# テスト
model.eval()

test_loss = 0.0
test_corrects = 0
with tqdm(enumerate(test_dataloader), total=len(test_dataloader)) as pbar_batch:
    for i, (inputs, labels, paths) in pbar_batch:
        inputs = inputs.to(device)
        labels = labels[:, 2].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        test_loss += loss.item() * inputs.size(0)
        test_corrects += torch.sum(preds == labels.data)
test_loss = test_loss / len(test_dataset)
test_acc = test_corrects / len(test_dataset)

# test_lossとtest_accをcpuに移し、numpy配列に変換
test_loss = np.array([test_loss])
test_acc = np.array([test_acc.cpu()])

# test_lossとtest_accをcsvに出力
import pandas as pd

df = pd.DataFrame({"test_loss": [test_loss], "test_acc": [test_acc]})
df.to_csv(result_path + "/test_result.csv", index=False)
