import numpy as np
from teili.tools.sorting import SortMatrix
import random

import matplotlib.pyplot as plt

trials = {'trial1': [True, 20, 20, 2],
         'trial2': [True, 20, 20, 1],
         'trial3': [True, 6, 6, 1]}

for values in trials.values():
    # Base matrix
    noise = values[0]
    n_rows = values[1] 
    n_cols = values[2] 
    px_cluster = values[3] 

    # First 3 plots of each trial:
    # 3 diagonals matrix
    a=np.zeros((n_rows, n_cols))
    b=np.zeros(n_cols)
    b[0:px_cluster]=1
    b[int(n_cols/2):int(n_cols/2) + px_cluster]=1
    for i in range(n_rows):
        a[i][np.where(b==1)]=15
        if noise:
            a[i][np.random.randint(n_cols, size=int(n_cols/2) - 1)] = 0
        b = np.roll(b, 1)

    sm3 = SortMatrix(ncols=n_cols, nrows=n_rows, matrix=a)
    plt.figure()
    plt.imshow(sm3.matrix)
    plt.title('original')
    plt.figure()
    plt.imshow(sm3.matrix[sm3.permutation, :])
    plt.title('sorted with permutation indices')
    plt.figure()
    plt.imshow(sm3.sorted_matrix)
    plt.title('sorted_matrix attribute')
    plt.show()

    # Last 4 plots of each trial:
    # 1 diagonal matrix
    a=np.zeros((n_rows, n_cols))
    b=np.zeros(n_cols)
    b[0:px_cluster]=1
    ids = [x for x in range(n_rows)]
    for i in ids:
        a[i][np.where(b==1)]=15
        if noise:
            a[i][np.random.randint(n_cols, size=int(n_cols/2) - 1)] = 0
        b = np.roll(b, 1)

    plt.figure()
    plt.imshow(a)
    plt.title('original')
    random.shuffle(ids)
    a = a[ids]
    sm1 = SortMatrix(ncols=n_cols, nrows=n_rows, matrix=a)
    plt.figure()
    plt.imshow(sm1.matrix)
    plt.title('original shuffled')
    plt.figure()
    plt.imshow(sm1.matrix[sm1.permutation, :])
    plt.title('sorted with permutation indices')
    plt.figure()
    plt.imshow(sm1.sorted_matrix)
    plt.title('sorted_matrix attribute')
    plt.show()
