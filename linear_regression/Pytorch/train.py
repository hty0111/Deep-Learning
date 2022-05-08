import torch
from torch import nn
from torch.utils.data import DataLoader
from config.trainConfig import TrainConfig
from data.dataloader import LinearDataset
from utils.visualize import Visualize
import os
import numpy as np
from matplotlib import pyplot as plt

def main():
    # ###########
    # config  
    # ###########  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    # print(torch.cuda.device_count())
    cfg = TrainConfig().getArgs()

    # ###########
    # load data  
    # ###########  
    dataset = LinearDataset(cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    # print(next(iter(dataloader)))
    
    # ###########
    # model
    # ########### 
    model = nn.Sequential(nn.Linear(len(cfg.true_w), 1)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)

    # ###########
    # train
    # ########### 
    for epoch in range(cfg.epochs):
        for feature, label in dataloader:
            # to device
            feature = feature.to(device)
            label = label.to(device)

            # forword
            predict = model(feature).squeeze(-1)
            loss = criterion(label, predict)

            # backword and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1) % 5 == 0:
            print(f"epoch: {epoch+1}, loss: {loss:.4f}")

    # ###########
    # save
    # ########### 
    save_path = os.path.join(cfg.model_root, f"linear_{epoch+1}epochs.pth")
    torch.save(model.state_dict(), save_path)

    # ###########
    # visualize
    # ########### 
    dataloader = DataLoader(dataset, batch_size=cfg.point_num, shuffle=False, num_workers=0)
    x, y = next(iter(dataloader))
    tensor_x = torch.from_numpy(np.array(x)).to(device)
    y_hat = model(tensor_x).detach().cpu().numpy()
    plt.scatter(x[:, 0], y, label="True")
    plt.scatter(x[:, 0], y_hat, label="Predict")
    plt.legend()
    plt.savefig(os.path.join(cfg.image_root, "prediction.png"))


if __name__=="__main__":
    main()
