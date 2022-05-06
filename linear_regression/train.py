import imp
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from config.trainConfig import TrainConfig
from data.dataloader import LinearDataset

def main():
    train_cfg = TrainConfig().getArgs()
    num_workers = train_cfg.num_workers

    # ###########
    # load data  
    # ###########  
    dataset = LinearDataset(train_cfg)
    dataloader = DataLoader(dataset, shuffle=True, num_workers=train_cfg.num_workers)
    # print(next(iter(dataloader)))
    
    # ###########
    # net
    # ########### 
    net = nn.Sequential(nn.Linear(2, 1)).to()

if __name__=="__main__":
    main()
