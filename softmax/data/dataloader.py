import numpy as np
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import os

class FMnistDataset(Dataset):
    def __init__(self, args):
        self.data_root = args.data_root

    def __len__(self):
        # return self.num
        pass

    def __getitem__(self, index: int):
        features, labels = self.generate_dataset()
        return features[index], labels[index]

    def generate_dataset(self):
        """fashion_mnist"""
        print(os.getcwd())
        # return x, y

