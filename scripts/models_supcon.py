from torch import nn
import torch.nn.functional as F

import timm
class SupConModel(nn.Module):

    def __init__(
        self, base_name: str, pretrained=False,
        in_channels: int=3, feat_dim: int=128
    ):
        """Initialize"""
        self.base_name = base_name
        super(SupConModel, self).__init__()

        # # prepare backbone
        if hasattr(timm.models, base_name):
            base_model = timm.create_model(
                base_name, num_classes=0, pretrained=pretrained, in_chans=in_channels)
            in_features = base_model.num_features
            print("load imagenet pretrained:", pretrained)
        else:
            raise NotImplementedError

        self.backbone = base_model
        print(f"{base_name}: {in_features}")

        # 参考
        # https://github.com/HobbitLong/SupContrast/blob/8d0963a7dbb1cd28accb067f5144d61f18a77588/networks/resnet_big.py#L174
        self.head = nn.Sequential(
                nn.Linear(in_features, in_features),
                nn.ReLU(inplace=True),
                nn.Linear(in_features, feat_dim)
            )

    def forward(self, x):
        """Forward"""
        feat = self.backbone(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat
    
    
class BasicModel(nn.Module):

    def __init__(
        self, base_name: str, pretrained=False,
        in_channels: int=3, out_dim: int=1
    ):
        """Initialize"""
        self.base_name = base_name
        super(BasicModel, self).__init__()

        # # prepare backbone
        if hasattr(timm.models, base_name):
            base_model = timm.create_model(
                base_name, num_classes=0, pretrained=pretrained, in_chans=in_channels)
            in_features = base_model.num_features
            print("load imagenet pretrained:", pretrained)
        else:
            raise NotImplementedError

        self.backbone = base_model
        print(f"{base_name}: {in_features}")

        self.head = nn.Linear(in_features, out_dim)

    def forward(self, x):
        """Forward"""
        h = self.backbone(x)
        h = self.head_cls(h)
        return h