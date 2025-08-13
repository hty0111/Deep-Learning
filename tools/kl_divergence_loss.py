import numpy as np

def kl_divergence(p, q, epsilon=1e-10):
    """
        p: [n_samples, n_classes] sum(p, axis=0) == 1
        q: [n_samples, n_classed] sum(q, axis=0) == 1
    """
    p = np.clip(p, epsilon, 1-epsilon)
    q = np.clip(q, epsilon, 1-epsilon)

    # loss = sum(p * log(p / q))
    kl = np.sum(p * np.log(p / q), axis=1)

    return np.mean(kl)

def kl_divergence_from_logits(logits, target_probs, epsilon=1e-10):
    pred_probs = softmax(logits)
    return kl_divergence(pred_probs, target_probs, epsilon)

def softmax(x):
    if x.ndim == 1:
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
    
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    