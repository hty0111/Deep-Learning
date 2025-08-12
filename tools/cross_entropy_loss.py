import numpy as np

def binary_cross_entropy(y_true, y_pred, epsilon=1e-10):
    """
        y_true: [batch_size, 1]
        y_pred: [batch_size, 1]
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # loss = - (y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    return np.mean(loss)

def categorical_cross_entropy(y_true, y_pred, epsilon=1e-10):
    """
        y_true: [batch_size, num_classed]
        y_pred: [batch_size, num_classed]
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # loss = -sum(y_true * log(y_pred)) / batch_size
    loss = -np.sum(y_true * np.log(y_pred), axis=1)

    return np.mean(loss)

    