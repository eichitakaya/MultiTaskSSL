"""
This file contains the dataloader for the RadImageNet dataset.
"""
# Imports
import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

# RadImageNet Dataset class
class RadImageNet(Dataset):
    """
    RadImageNet dataset class.
    The format of output is (image, label).
    Label is 3-dim vector, where each element is corresponding to the label of modalities, anatomy, and disease. 
    Example: [0, 1, 2] means the image is from CT, Abdomen, and Abdominal Aortic Aneurysm.
    """
    def __init__(self, root, transform=None, target_transform=None):
        """
        Constructor for RadImageNet dataset class.
        This does not load the data into memory.
        
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        # RadImageNetのパスからモダリティ、解剖学、疾患の辞書を作成
        self.modality_dict = {}
        modalities = glob.glob(root + "/*")
        for i, modality in enumerate(modalities):
            self.modality_dict[modality.split("/")[-1]] = i
        self.anatomy_dict = {}
        anatomies = glob.glob(root + "/*/*")
        for i, anatomy in enumerate(anatomies):
            self.anatomy_dict[anatomy.split("/")[-1]] = i
        self.disease_dict = {}
        diseases = glob.glob(root + "/*/*/*")
        for i, disease in enumerate(diseases):
            self.disease_dict[disease.split("/")[-1]] = i
            
        # 全ての画像のパスを取得
        self.paths = glob.glob(root + "/*/*/*/*")

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Returns the item at the given index.
        Loads the image, label and img_path at the given index.
        Label is 3-dim vector, where each element is corresponding to the label of modalities, anatomy, and disease.
        """
        # 画像の読み込み
        img_path = self.paths[idx]
        img = Image.open(img_path)
        # 画像の前処理
        if self.transform:
            img = self.transform(img)
        img = np.array(img)
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)

        # ラベルの作成
        label = np.zeros(3)
        # モダリティのラベル
        modality = img_path.split("/")[-4]
        label[0] = self.modality_dict[modality]
        # 解剖学のラベル
        anatomy = img_path.split("/")[-3]
        label[1] = self.anatomy_dict[anatomy]
        # 疾患のラベル
        disease = img_path.split("/")[-2]
        label[2] = self.disease_dict[disease]
        # ラベルをint型に変換
        label = label.astype(np.int64)
        # ラベルをTensor型に変換
        label = torch.from_numpy(label)
        return img, label, img_path


# RadImageNetを継承した教師あり対照学習用のデータセットクラス
class RadImageNetForSupCon(RadImageNet):
    """
    RadImageNet dataset class for supervised contrastive learning.
    The format of output is (image, label).
    Label is 3-dim vector, where each element is corresponding to the label of modalities, anatomy, and disease. 
    Example: [0, 1, 2] means the image is from CT, Abdomen, and Abdominal Aortic Aneurysm.
    """
    def __init__(self, root, transform=None, target_transform=None):
        """
        Constructor for RadImageNet dataset class.
        This does not load the data into memory.
        
        """
        super(RadImageNetForSupCon, self).__init__(root, transform, target_transform)
    
    def __getitem__(self, idx):
        """
        Returns the item at the given index.
        Loads the image, label and img_path at the given index.
        Label is 3-dim vector, where each element is corresponding to the label of modalities, anatomy, and disease.
        """
        # 画像の読み込み
        img_path = self.paths[idx]
        img = Image.open(img_path)
        img = self.transform(img)
        # ここで2枚分のリストになっている

        # ラベルの作成
        label = np.zeros(3)
        # モダリティのラベル
        modality = img_path.split("/")[-4]
        label[0] = self.modality_dict[modality]
        # 解剖学のラベル
        anatomy = img_path.split("/")[-3]
        label[1] = self.anatomy_dict[anatomy]
        # 疾患のラベル
        disease = img_path.split("/")[-2]
        label[2] = self.disease_dict[disease]
        # ラベルをint型に変換
        label = label.astype(np.int64)
        # ラベルをTensor型に変換
        label = torch.from_numpy(label)
        return img, label, img_path

# test
if __name__ == "__main__":
    # dataset
    dataset = RadImageNet(root="../../Medical_AI/data/RadImageNet")
    # print(dataset.__len__())
    # print(dataset.modality_dict)
    # print(dataset.anatomy_dict)
    # print(dataset.disease_dict)

   
    # dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    # データを取り出す
    for i, (images, labels, img_paths) in enumerate(dataloader):
        #print(images.shape)
        #print(labels[:,1])
        #print(img_paths)
        print(len(dataloader))
        break
    