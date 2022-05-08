import torch
import numpy as np
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

class LinearDataset(Dataset):
    def __init__(self, args):
        self.w = args.true_w
        self.b = args.true_b
        self.num = args.point_num

    def __len__(self):
        return self.num

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]

    def generate_dataset(self):
        """y = x * w + b"""
        w = torch.tensor(np.array(self.w).T, dtype=torch.float32)
        b = torch.tensor(np.array(self.b), dtype=torch.float32)
        x = torch.normal(0, 1, (self.num, len(w)))
        y = torch.matmul(x, w) + b + torch.normal(0, 1, (x.shape[0],))
        return x, y

if __name__=="__main__":
    dataset = LinearDataset()
