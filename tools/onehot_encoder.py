'''
Description: 
version: v1.0
Author: HTY
Date: 2025-08-12 19:08:59
'''
import numpy as np

def one_hot(labels):
    n_samples = len(labels)
    n_classes = max(labels) + 1

    onehot_matrix = np.zeros((n_samples, n_classes), dtype=int)
    onehot_matrix[np.arange(n_samples), labels] = 1

    return onehot_matrix
