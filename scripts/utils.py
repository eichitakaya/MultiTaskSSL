from PIL import Image
import numpy as np

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

# (bsz, channels, height, width)の1枚目の画像を保存する関数
def save_image(tensor, filename):
    # テンソルを画像に変換
    img = tensor.clone().detach()
    img = img[0]
    img = img.cpu().numpy().transpose(1, 2, 0)
    img = img * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    # 画像を保存
    img.save(filename)
    return
