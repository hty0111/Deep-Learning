import torch
import matplotlib.pyplot as plt
import numpy as np
import random
# from d2l import torch as d2l
from torch.utils import data
from torch import nn


def synthetic(w, b, num_examples):
    """生成 y = Xw + b + 噪声"""
    X = torch.normal(0, 1, size=(num_examples, len(w)))  # 生成均值为0，方差为1的随机数据
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)     # 生成噪声
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
num_examples = 1000
features, labels = synthetic(true_w, true_b, num_examples)


def load_data(data_arrays, batch_size, is_train=True):
    """构造一个pytorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_data((features, labels), batch_size)


if __name__ == '__main__':
    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)     # 初始化参数，重写参数值
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss()     # 损失函数是均方误差

    trainer = torch.optim.SGD(net.parameters(), lr=0.01)    # 定义优化算法

    num_epochs = 10

    for epoch in range(num_epochs):
        for X, y in data_iter:
            Loss = loss(net(X), y)
            trainer.zero_grad()
            Loss.backward()
            trainer.step()

    w = net[0].weight.data
    print(f'error of w: ', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print(f'error of b: ', true_b - b)

    # plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
    plt.show()
