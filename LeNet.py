import torch
import torch.nn as nn
import d2l.torch as d2l
import numpy as np


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_acc_on_gpu(net, data_iter, device=None):
    """使用GPU计算精度"""
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(3)
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)


if __name__ == '__main__':
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    learning_rate = 0.9
    num_epochs = 10
    device = d2l.try_gpu()

    net = nn.Sequential(
        Reshape(),
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), padding=2),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        nn.Sigmoid(),
        nn.Linear(84, 10)
    )
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    loss = nn.CrossEntropyLoss()
    num_batches = len(train_iter)
    train_loss_vsl = []
    train_acc_vsl = []
    test_acc_vsl = []
    train_loss = 0
    train_acc = 0
    test_acc = 0
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            test_acc = evaluate_acc_on_gpu(net, test_iter)
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                train_loss_vsl.append(train_loss)
                train_acc_vsl.append(train_acc)
                test_acc_vsl.append(test_acc)
    print(f'train_loss {train_loss:.3f}, train_acc {train_acc:.3f}, test_acc {test_acc:.3f}')


    # x = torch.rand(size=(1, 1, 28, 28))
    # for layer in net:
    #     x = layer(x)
    #     print(layer.__class__.__name__, 'shape:\t', x.shape)

