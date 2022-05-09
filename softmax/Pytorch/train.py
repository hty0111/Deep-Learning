import torch
from torch import nn
import torchvision
from torchvision import transforms
from config.baseConfig import BaseConfig
import os
import numpy as np
from matplotlib import pyplot as plt
from utils.visualize import Visualize
from models.MLP import MLP
from tqdm import tqdm

def main():
    # ###########
    # config  
    # ###########  
    cfg = BaseConfig().getArgs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    # print(torch.cuda.device_count())
    
    # ###########
    # load data  
    # ###########  
    train_dataset = torchvision.datasets.FashionMNIST(root=cfg.data_root, train=True, transform=transforms.ToTensor(), download=True)

    # print(len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    # x, y = next(iter(train_loader))
    # print(x.shape, y.shape)
    # ###########
    # model
    # ########### 
    model = MLP(28*28, 500, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # ###########
    # train
    # ########### 
    for epoch in range(cfg.epochs):
        for i, (image, label) in enumerate(train_loader):
            image = image.reshape(-1, 28*28).to(device)
            label = label.long().to(device)

            predict = model(image)
            loss = criterion(predict, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f"Epoch: {epoch+1}/{cfg.epochs}, Step: {i+1}/{len(train_loader)}, Loss: {loss}")

    # ###########
    # save
    # ########### 
    save_path = os.path.join(cfg.model_root, f"MLP_{epoch+1}epochs.pth")
    torch.save(model.state_dict(), save_path)


if __name__=="__main__":
    main()
