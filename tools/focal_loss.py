'''
Description: 
version: v1.0
Author: HTY
Date: 2025-08-13 21:52:22
'''
import numpy as np

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0, epsilon=1e-10):
    """
        alpha: 类别平衡因子
        gamma: 聚焦参数，增强对难分类样本的关注
    """

    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    p_t = np.where(y_true == 1, y_pred, 1 - y_pred)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    loss = -alpha_t * (1 - p_t)**gamma * np.log(p_t)

    return np.mean(loss)
