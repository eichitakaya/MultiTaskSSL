# 学習済みのモデルを用いて、下流タスクの評価を行うスクリプト

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import downstream_small_dataset as dd
import argparse
import copy


# modelとdatasetを引数に取り、評価を行う関数
def downstream(model_init, dataset):
    if dataset == "thyroid":
        dataset = dd.ThyroidDataset
    if dataset == "breast":
        dataset = dd.BreastUSDataset
    if dataset == "acl":
        dataset = dd.ACLDataset
    # 5foldのtrain/val/testを行う
    preds_labels_all = []
    for i in range(5):
        # データセットの読み込み
        train_dataset = dataset(i+1, "train", transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
        val_dataset = dataset(i+1, "val", transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
        test_dataset = dataset(i+1, "test", transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
        # データローダーの作成
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
        # モデルの定義
        model = copy.deepcopy(model_init)
        # 損失関数の定義
        criterion = nn.CrossEntropyLoss()
        # 最適化手法の定義
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # 学習率のスケジューラーの定義
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        # モデルの定義
        model = model.to("cuda:0")
        
        # 学習ループ
        # testは、valのlossが最小のモデルを使用
        best_val_loss = 10000
        model.train()
        preds_labels_all = []
        
        for epoch in range(10):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_dataloader:
                inputs = inputs.to("cuda:0")
                labels = labels.to("cuda:0")
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print("epoch: {}, loss: {}".format(epoch, running_loss))
            scheduler.step()
            # validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs = inputs.to("cuda:0")
                    labels = labels.to("cuda:0")
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                print("val_loss: ", val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model

        # test
        # 1である確率をlabelsと結合し、numpy形式でreturn
        test_loss = 0.0
        test_corrects = 0
        preds_fold = []
        labels_fold = []
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.to("cuda:0")
                labels = labels.to("cuda:0")
                outputs = best_model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                test_corrects += torch.sum(preds == labels.data)
                preds_fold.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                labels_fold.extend(labels.cpu().numpy())
        test_loss = test_loss / len(test_dataset)
        preds_labels_all.append([preds_fold, labels_fold])
        
    return preds_labels_all

def roc_curve(preds, labels):
    # ROC曲線を描画
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds)
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label="ROC curve (area = %.2f)" % auc)
    plt.legend()
    plt.title("ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.show()
 
# コマンドライン引数の処理
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--model_path", type=str)
args = parser.parse_args()

# モデルの定義と学習済みパラメータの読み込み
# vitを使う
model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
# モデルの最終層の出力ユニットを165に変更
model.fc = nn.Linear(in_features=512, out_features=165, bias=True)

# モデルの評価
result = downstream(model, dataset=args.dataset)

# 評価結果の集計と保存
# result自体をcsvに保存
# acc, auc, sensitivity, specificityを計算し、csvに保存
# roc_curveを各foldについて描画、平均のROC曲線も重ねて描画
# 保存先はmodel_pathと同じディレクトリにresult.csv, roc_curve.png