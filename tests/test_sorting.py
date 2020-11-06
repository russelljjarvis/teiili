# TODO put this on test_tools.py
import numpy as np
from teili.tools.sorting import SortMatrix
import random
import copy

import matplotlib.pyplot as plt

# Base matrix
n_rows = 50
n_cols = 50
diag_width = 3
max_weight = 15
noise = True 
noise_probability = 0.05

test_matrix = np.zeros((n_rows, n_cols))
if max_weight > 1:
    test_matrix = (max_weight*np.random.rand(n_rows, n_cols)).astype(int)
    test_matrix = np.clip(test_matrix, 0, int(max_weight/2))
else:
    test_matrix = (max_weight*np.random.rand(n_rows, n_cols))
    test_matrix = np.clip(test_matrix, 0, max_weight)
    

# Construct matrix
ids = [x for x in range(n_rows)]
for central_id in ids:
    # Add values
    add_to_ids = np.arange(central_id-diag_width, central_id+diag_width+1)
    values_to_add = np.array([max_weight for _ in range(2*diag_width+1)])
    # Remove values outside dimensions of matrix
    values_to_add = np.delete(values_to_add, np.append(
        np.where(add_to_ids<0), np.where(add_to_ids>n_rows-1)))
    add_to_ids = np.delete(add_to_ids, np.append(
        np.where(add_to_ids<0), np.where(add_to_ids>n_rows-1)))
    try:
        test_matrix[central_id][add_to_ids] = values_to_add
    except IndexError:
        continue

# Manually add noise
if noise:
    noise_ind = np.where(np.random.rand(n_rows, n_cols) < noise_probability)
    test_matrix[noise_ind] = 0.3


# Shuffle matrix
shuffled_matrix_tmp = np.zeros((n_rows, n_cols)) # this is to test recurrent matrices which need to be shuffled along rows and columns
shuffled_matrix = np.zeros((n_rows, n_cols))
inds = np.arange(n_cols, dtype='uint32')  # 32 bits is enough for numbers up to about 4 billion
np.random.shuffle(inds)
shuffled_matrix_tmp[:, inds] = copy.deepcopy(test_matrix)
shuffled_matrix[inds, :] = shuffled_matrix_tmp

sm1 = SortMatrix(ncols=n_cols, nrows=n_rows, axis=1, matrix=copy.deepcopy(shuffled_matrix))


plt.figure(figsize=(11,8))
plt.subplot(2,2,1)
plt.imshow(test_matrix)
plt.title('Original matrix')
plt.colorbar()

plt.subplot(2,2,2)
plt.imshow(shuffled_matrix)
plt.title('Shuffled matrix')
plt.colorbar()

plt.subplot(2,2,3)
plt.imshow(shuffled_matrix[:, sm1.permutation])
plt.title('Sorted with permutation indices')
plt.colorbar()

plt.subplot(2,2,4)
plt.imshow(sm1.sorted_matrix)
plt.title('Sorted matrix')
plt.colorbar()

plt.show()
