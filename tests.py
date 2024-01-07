from PIL import Image
import os
import json
import numpy as np

if __name__ == '__main__':
    n = 6
    y = np.array([2, 4, 1, 0, 3, 5])
    categorical = np.zeros((len(y), n))

    X_train = np.random.rand(1000, 10)
    y_train = np.random.randint(0, 3, 1000)
    j = 10

