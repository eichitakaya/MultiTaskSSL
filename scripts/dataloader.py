"""
This file contains the dataloader for the RadImageNet dataset.
"""
# Imports
import os
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
    """
    def __init__(self, root, transform=None, target_transform=None):
        """
        Constructor for the RadImageNet dataset class.
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.imgs = ImageFolder(root=root, transform=transform, target_transform=target_transform)

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Returns the item at the given index.
        """
        return self.imgs[idx]