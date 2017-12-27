from brian2 import implementation, check_units, ms, exp, mean, diff, declare_types
import numpy as np


# function that calculates 1D index from 2D index
# same as np.ravel_multi_index((x,y),(n2dNeurons,n2dNeurons))
@implementation('numpy', discard_units=True)
@check_units(x=1, y=1, n2dNeurons=1, result=1)
def xy2ind(x, y, n2dNeurons):
    """Given a pair of x, y (pixel) coordinates this function
    will return an index that correspond to a flattened pixel array

    Args:
        x (int, required): x-coordinate
        y (int, rquired): y-coordinate
        n2dNeurons (TYPE): Longest edge of the original array

    Returns:
        ind (int): Converted index (e.g. flattened array)
    """
    if isinstance(x, np.ndarray):
        return x + (y * n2dNeurons)
    else:
        return int(x) + int(y) * n2dNeurons

# function that calculates 2D index from 1D index
# please note that the total number of neurons in the square field is n2dNeurons**2
#@implementation('numpy', discard_units=True)


@implementation('cpp', '''
    int ind2x(int ind,int n2dNeurons) {
    return ind / n2dNeurons;
    }
     ''')
@declare_types(ind='integer', n2dNeurons='integer', result='integer')
@check_units(ind=1, n2dNeurons=1, result=1)
def ind2x(ind, n2dNeurons):
    """Given an index of an array this function will provide
    you with the corresponding x coordinate

    Args:
        ind (int, required): index of flattened array that should be converted back to pixel cooridnates
        n2dNeurons (int, required): Longest edge of the original array

    Returns:
        x (int): The x coordinate of the respective index in the unflattened array
    """
    return np.floor_divide(np.round(ind), n2dNeurons)


@implementation('cpp', '''
    int ind2y(int ind,int n2dNeurons) {
    return ind % n2dNeurons;
    }
     ''')
@declare_types(ind='integer', n2dNeurons='integer', result='integer')
@check_units(ind=1, n2dNeurons=1, result=1)
def ind2y(ind, n2dNeurons):
    """Given an index of an array this function will provide
    you with the corresponding y coordinate

    Args:
        ind (int, required): index of flattened array that should be converted back to pixel cooridnates
        n2dNeurons (int, required): Longest edge of the original array

    Returns:
        y (int): The y coordinate of the respective index in the unflattened array
    """
    ret = np.mod(np.round(ind), n2dNeurons)
    return ret


@implementation('numpy', discard_units=True)
@check_units(ind=1, n2dNeurons=1, result=1)
def ind2xy(ind, n2dNeurons):
    """Given an index of an array this function will provide
    you with the corresponding x and y coordinate of the original array

    Args:
        ind (int, required): index of flattened array that should be converted back to pixel cooridnates
        n2dNeurons (int, required): Longest edge of the original array

    Returns:
        tuple (x, y): The corresponding x, y coordinates
    """
    if type(n2dNeurons) == tuple:
        return np.unravel_index(ind, (n2dNeurons[0], n2dNeurons[1]))
    elif type(n2dNeurons) == int:
        return np.unravel_index(ind, (n2dNeurons, n2dNeurons))


# function that calculates distance in 2D field from 2 1D indices
@implementation('cpp', '''
    float fdist2d(int i, int j, int n2dNeurons) {
    int ix = i / n2dNeurons;
    int iy = i % n2dNeurons;
    int jx = j / n2dNeurons;
    int jy = j % n2dNeurons;
    return sqrt(pow((ix - jx),2) + pow((iy - jy),2));
    }
     ''')
@declare_types(i='integer', j='integer', n2dNeurons='integer', result='float')
@check_units(i=1, j=1, n2dNeurons=1, result=1)
def fdist2d(i, j, n2dNeurons):
    """Summary

    Args:
        i (TYPE): Description
        j (TYPE): Description
        n2dNeurons (TYPE): Description

    Returns:
        TYPE: Description
    """
    (ix, iy) = ind2xy(i, n2dNeurons)
    (jx, jy) = ind2xy(j, n2dNeurons)
    return np.sqrt((ix - jx)**2 + (iy - jy)**2)
