import os
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
import torchvision
from torchvision import transforms
from config.baseConfig import BaseConfig
from models.MLP import MLP
from utils.label import Label
from utils.metrics import Metrics
from utils.visualize import Visualize

def main():
    # ###########
    # config  
    # ###########  
    cfg = BaseConfig().getArgs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device, torch.cuda.device_count())

    # ###########
    # load data  
    # ###########  
    train_dataset = torchvision.datasets.FashionMNIST(root=cfg.data_root, train=True, transform=transforms.ToTensor(), download=True)
    # print(len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    x, y = next(iter(torch.utils.data.DataLoader(train_dataset, batch_size=20)))
    label = Label(cfg)
    label.show_labels(x.reshape(20, 28, 28), 2, 10, titles=label.get_fashion_mnist_labels(y))

    # ###########
    # model
    # ########### 
    model = MLP(28*28, 500, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # ###########
    # train
    # ###########
    metrics = Metrics(cfg)
    min_loss = np.inf
    for epoch in range(cfg.epochs):
        for i, (images, labels) in enumerate(train_loader):
            # to device
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.long().to(device)

            # forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # loss and accuracy
            with torch.no_grad():
                metrics.update(loss.item(), outputs, labels)
        
        epoch_loss, epoch_acc = metrics.get_epoch_metrics()
        if epoch_loss < min_loss:
            min_loss = epoch_loss
        print(f"Epoch: {epoch+1} | Min loss: {min_loss} | Loss: {epoch_loss} | Acc: {epoch_acc}")
        metrics.clear_metrics()

    # ###########
    # save
    # ########### 
    save_path = os.path.join(cfg.model_root, f"MLP_{epoch+1}epochs.pth")
    torch.save(model.state_dict(), save_path)

    # ###########
    # visualize
    # ###########   
    loss, acc = metrics.get_metrics()
    vis = Visualize(cfg.image_root)
    vis.plot_train(loss, acc)

if __name__=="__main__":
    main()
