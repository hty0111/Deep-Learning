import torch
import numpy as np
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

class LinearDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.w = self.args.true_w
        self.b = self.args.true_b
        self.num = self.args.point_num
        self.num_workers = self.args.num_workers
        self.features, self.labels = self.generate_dataset(self.w, self.b, self.num)

    def __len__(self):
        return self.num

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]

    def generate_dataset(self, w, b, num):
        """y = x * w + b"""
        w = torch.tensor(np.array(w).T, dtype=torch.float32)
        b = torch.tensor(np.array(b), dtype=torch.float32)
        x = torch.normal(0, 1, (num, len(w)))
        y = torch.matmul(x, w) + b + torch.normal(0, 1, (x.shape[0],))
        return x, y

if __name__=="__main__":
    dataset = LinearDataset()
