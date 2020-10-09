import numpy as np
from teili.tools.sorting import SortMatrix
import random

import matplotlib.pyplot as plt

trials = {#'trial1': [True, 3, 3, 1, False, .5, .7],
          #'trial2': [True, 4, 4, 1, False, .5, .7],
          #'trial3': [True, 5, 5, 1, False, .5, .7],
          'trial4': [True, 6, 6, 1, False, .5, .7]}
          #'trial2': [False, 20, 20, 0, False, .5, .7],
          #'trial2': [False, 20, 20, 1, False, .5, .7],
          #'trial3': [False, 20, 20, 2, False, .5, .7]}

for values in trials.values():
    # Base matrix
    noise = values[0]
    n_rows = values[1] 
    n_cols = values[2] 
    diag_width = values[3]
    px_grad = values[4]
    min_val = values[5]
    max_val = values[6]
    if n_rows < (2*diag_width+1):
        print('Matrix too small for data')
        import sys;sys.exit()

    # First 3 plots of each trial:
    # 3 diagonals matrix
    test_matrix = np.zeros((n_rows, n_cols))
    aux_vec = np.zeros(n_cols)
    aux_vec[0] = 1
    aux_vec[int(n_cols/2)] = 1
    for i in range(n_rows):
        # Define where to add values
        central_ids = np.where(aux_vec==1)[0]
        add_to_ids = np.append(
            np.arange(central_ids[0]-diag_width, central_ids[0]+diag_width+1),
            np.arange(central_ids[1]-diag_width, central_ids[1]+diag_width+1)
            )
        # Create gradient around central pixels
        temp_values = np.linspace(min_val, max_val, num=diag_width+1)
        if not px_grad:
            temp_values[:] = max_val
        values_to_add = np.concatenate([temp_values, temp_values[::-1][1:],
                                        temp_values, temp_values[::-1][1:]])
        # Remove values outside dimensions of matrix
        values_to_add = np.delete(values_to_add, np.append(
            np.where(add_to_ids<0), np.where(add_to_ids>n_rows-1)))
        add_to_ids = np.delete(add_to_ids, np.append(
            np.where(add_to_ids<0), np.where(add_to_ids>n_rows-1)))
        test_matrix[i][add_to_ids]=values_to_add
        if noise:
            test_matrix[i][np.random.randint(n_cols, size=int(n_cols/2) - 1)] = 0
        aux_vec = np.roll(aux_vec, 1)

    sm3 = SortMatrix(ncols=n_cols, nrows=n_rows, matrix=test_matrix)
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
    test_matrix=np.zeros((n_rows, n_cols))
    ids = [x for x in range(n_rows)]
    for central_id in ids:
        # Define where to add values
        add_to_ids = np.arange(central_id-diag_width, central_id+diag_width+1)
        # Create gradient around central pixel
        temp_values = np.linspace(min_val, max_val, num=diag_width+1)
        if not px_grad:
            temp_values[:] = max_val
        values_to_add = np.append(temp_values, temp_values[::-1][1:])
        # Remove values outside dimensions of matrix
        values_to_add = np.delete(values_to_add, np.append(
            np.where(add_to_ids<0), np.where(add_to_ids>n_rows-1)))
        add_to_ids = np.delete(add_to_ids, np.append(
            np.where(add_to_ids<0), np.where(add_to_ids>n_rows-1)))
        test_matrix[central_id][add_to_ids]=values_to_add
        if noise:
            test_matrix[central_id][np.random.randint(n_cols, size=int(n_cols/2) - 1)] = 0

    plt.figure()
    plt.imshow(test_matrix)
    plt.title('original')
    random.shuffle(ids)
    test_matrix = test_matrix[ids]
    sm1 = SortMatrix(ncols=n_cols, nrows=n_rows, matrix=test_matrix)
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
