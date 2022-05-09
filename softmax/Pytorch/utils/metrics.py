import numpy as np
import torch

class Metrics(object):
    def __init__(self, cfg) -> None:
        self.epoch = cfg.epochs
        self.iter_loss = []
        self.iter_acc = []
        self.epoch_loss = []
        self.epoch_acc = []
        self.correct = 0

    def update(self, iter_loss, outputs, labels):
        self.iter_loss.append(iter_loss)
        self.iter_acc.append(self.get_acc(outputs, labels))

    def get_epoch_metrics(self):
        loss = np.mean(self.iter_loss)
        acc = np.mean(self.iter_acc)
        self.epoch_loss.append(loss)
        self.epoch_acc.append(acc)
        return loss, acc

    def get_metrics(self):
        return self.epoch_loss, self.epoch_acc

    def get_acc(self, outputs, labels):
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        return correct / total   

    def clear_metrics(self):
        self.iter_loss = []
        self.iter_acc = []
