import torch
from matplotlib import pyplot as plt
import os

class Visualize():
    def __init__(self, cfg) -> None:
        self.image_root = cfg.image_root

    def get_fashion_mnist_labels(self, labels):
        """返回Fashion-MNIST数据集的文本标签"""
        text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [text_labels[int(i)] for i in labels]


    def show_labels(self, imgs, num_rows, num_cols, titles=None, scale=1.5):
        figsize = (num_cols*scale, num_rows*scale)
        _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten()
        for i, (ax, img) in enumerate(zip(axes, imgs)):
            if torch.is_tensor(img):
                # 图片张量
                ax.imshow(img.numpy())
            else:
                # PIL图片
                ax.imshow(img)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if titles:
                ax.set_title(titles[i])
        plt.savefig(os.path.join(self.image_root, "labels.png"))
        return axes