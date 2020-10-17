import numpy as np

def clip_gradients(in_grads, clip=1):
    return np.clip(in_grads, -clip, clip)

def sigmoid(X):
    return 1.0 / (1 + np.exp(-X))

