# TODO put this on test_tools.py
import numpy as np
from teili.tools.sorting import SortMatrix
import random
import copy

import matplotlib.pyplot as plt

# Base matrix
n_rows = 8
n_cols = 8
diag_width = 1

test_matrix = np.zeros((n_rows, n_cols))
test_matrix = (8*np.random.rand(n_rows, n_cols)).astype(int)
test_matrix = np.clip(test_matrix, 0, 15)

# Construct matrix
ids = [x for x in range(n_rows)]
for central_id in ids:
    # Add values
    add_to_ids = np.arange(central_id-diag_width, central_id+diag_width+1)
    values_to_add = np.array([15 for _ in range(2*diag_width+1)])
    # Remove values outside dimensions of matrix
    values_to_add = np.delete(values_to_add, np.append(
        np.where(add_to_ids<0), np.where(add_to_ids>n_rows-1)))
    add_to_ids = np.delete(add_to_ids, np.append(
        np.where(add_to_ids<0), np.where(add_to_ids>n_rows-1)))
    test_matrix[central_id][add_to_ids] = values_to_add

# Manually add noise
test_matrix[0,1] = 0.
test_matrix[1,1] = 0.
test_matrix[3,3] = 0.
test_matrix[4,4] = 0.
test_matrix[5,4] = 0.
test_matrix[7,6] = 0.

# Shuffle matrix
shuffled_matrix = np.zeros((n_rows, n_cols))
#rnd_ind = np.asarray([3,1,0,2])
rnd_ind = np.asarray([7,5,1,4,3,0,2,6])
shuffled_matrix[:, rnd_ind] = test_matrix
sm1 = SortMatrix(ncols=n_cols, nrows=n_rows, axis=1, matrix=copy.deepcopy(shuffled_matrix))

plt.figure()
plt.imshow(test_matrix)
plt.title('Original matrix')
plt.colorbar()

plt.figure()
plt.imshow(shuffled_matrix)
plt.title('Shuffled matrix')
plt.colorbar()

plt.figure()
plt.imshow(shuffled_matrix[:, sm1.permutation])
plt.title('Sorted with permutation indices')
plt.colorbar()
plt.show()
