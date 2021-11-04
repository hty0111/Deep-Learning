import torch
import torch.nn as nn
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


def get_labels(labels):
    """Return text labels for the Fashion-MNIST dataset."""
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def init_weights(m):
    """初始化权重，均值为0，标准差为0.01"""
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


if __name__ == '__main__':
    batch_size = 256
    num_workers = 4     # 进程数

    # (1) 加载Fashion-MNIST数据到内存

    # 将图片格式转为tensor和[channel, w, h]
    # 将图像数据从PIL类型变换成32位浮点数格式，并除以255使得所有像素的数值均在0到1之间
    trans = [transforms.ToTensor()]
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data",
                                                    train=True,
                                                    transform=trans,
                                                    download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data",
                                                   train=False,
                                                   transform=trans,
                                                   download=True)
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers)
    test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=num_workers)

    # (2) 初始化模型参数

    learning_rate = 0.1
    num_epochs = 10
    num_genre = 10
    h = 28              # 高度像素
    w = 28              # 宽度像素
    # PyTorch不会隐式地调整输入的形状。因此我们在线性层前定义了展平层（flatten），来调整网络输入的形状
    net = nn.Sequential(nn.Flatten(), nn.Linear(h * w, num_genre))
    # 初始化权重
    net.apply(init_weights)
    # 定义损失函数
    loss = nn.CrossEntropyLoss()
    # 定义优化器
    updater = torch.optim.SGD(net.parameters(), lr=learning_rate)

    # (3) 训练

    train_metric = np.zeros(3)      # 存放训练损失总和、训练准确度总和、样本数
    test_metric = np.zeros(2)       # 存放验证集准确度总和、样本数

    for epoch in range(num_epochs):
        if isinstance(net, torch.nn.Module):
            net.train()         # 设置为训练模式
        train_metric = np.zeros(3)

        for X, y in train_iter:
            y_hat = net(X)      # 计算训练的估计值
            l = loss(y_hat, y)  # 计算损失函数
            updater.zero_grad() # 梯度清零
            l.backward()        # 反向传播
            updater.step()      # 更新参数
            train_metric += np.array([float(l) * len(y), accuracy(y_hat, y), y.numel()])

        if isinstance(net, torch.nn.Module):
            net.eval()
        test_metric = np.zeros(2)

        with torch.no_grad():
            for X, y in test_iter:
                test_metric += np.array([accuracy(net(X), y), y.numel()])

    train_loss = train_metric[0] / train_metric[2]
    train_accuracy = train_metric[1] / train_metric[2]
    test_accuracy = test_metric[0] / test_metric[1]
    print(train_loss, train_accuracy, test_accuracy)

    # (4) 可视化

    for X, y in test_iter:
        break
    true_label = get_labels(y)
    pre_label = get_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(true_label, pre_label)]

    scale = 2
    figure_size = (h * scale, w * scale)
    DPI = 80
    num_rows = 2
    num_cols = 9
    num_fig = num_cols * num_rows       # 图片数量
    _, ax = plt.subplots(num_rows, num_cols, figsize=figure_size)
    ax = ax.flatten()
    imgs = X[0:num_fig].reshape(num_fig, h, w)
    for index, (a, img) in enumerate(zip(ax, imgs)):
        a.imshow(img)
        a.axes.get_xaxis().set_visible(False)
        a.axes.get_yaxis().set_visible(False)
        a.set_title(titles[index])
    plt.show()

