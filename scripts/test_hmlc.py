# hmlcの動作検証
# pytorchのtensor型で、(10, 2, 10)のshapeを持つtensorを作成してfeaturesとする
# このfeaturesに対して、(10)のshapeを持つtensorを作成してlabelとする
# featuresとlabelをhmlcに入力して、forward関数を実行する

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
from loss import HMLC, SupConLoss

# featuresのshapeは(10, 2, 10)
#features = torch.randn(10, 2, 128)
#print(features.shape)

# labelのshapeは(10)
#label = torch.randint(0, 5, (10, 3))
#print(label)

bsz = 10
feature_dim = 128
n_views = 2
num_classes_per_layer = [3, 10, 25]
features = torch.rand(bsz, n_views, feature_dim)
labels = torch.cat([torch.randint(0, cls, (bsz, 1)) for cls in num_classes_per_layer], dim=1)
print(labels)


# hmlcの初期化
loss = HMLC(loss_type="hmce")

# hmlcのforward関数を実行
output = loss(features, labels)
#print(output)