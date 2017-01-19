from brian2 import *
import numpy as np


def setParams(briangroup, params, ndargs=None, debug=False):
    for par in params:
        if hasattr(briangroup, par):
            if ndargs is not None and par in ndargs:
                if ndargs[par] is None:
                    setattr(briangroup, par, params[par])
                else:
                    setattr(briangroup, par, ndargs[par])
            else:
                setattr(briangroup, par, params[par])
    if debug:
        states = briangroup.get_states()
        print '\n' 
        print '-_-_-_-_-_-_-_', '\n', 'Parameters set'
        for key in states.keys():
            if key in params:
                print key, states[key]
        print '----------'


# function that calculates 1D index from 2D index
@implementation('numpy', discard_units=True)
@check_units(x0=1,x1=1,n2dNeurons=1,result=1)
def xy2ind(x0,x1,n2dNeurons):
    return int(x0)+int(x1)*n2dNeurons

# function that calculates 2D index from 1D index
@implementation('numpy', discard_units=True)
@check_units(ind=1,n2dNeurons=1,result=1)
def ind2xy(ind,n2dNeurons):
    ret = (np.mod(np.round(ind),n2dNeurons), np.floor_divide(np.round(ind),n2dNeurons))     
    return ret

# from Brian2 Equations class
def replaceEqVar(eq , varname, replacement, debug=False):
    "replaces variables in equations like brian 2, helper for replaceConstants"
    if isinstance(replacement, str):
        eq = eq.replace(varname,replacement)
    else:
        eq = eq.replace(varname,'(' + repr(replacement) + ')')
            
    if debug:
        print('replaced ' + str(varname) + ' by ' + str(repr(replacement)))
    return (eq)


def replaceConstants(equation,replacedict, debug=False):
    "replaces constants in equations and deletes the respective definitions, given a dictionary of replacements"
    for key in replacedict:
        if replacedict[key] is not None:            
            # delete line from model eq
            neweq = ''
            firstline = True
            for line in equation.splitlines():
                if not all([kw in line for kw in [key,'(constant)']]):
                    if firstline:
                        neweq = neweq + line
                        firstline = False
                    else:
                        neweq = neweq  +'\n' + line
                else:
                    print('deleted ' + str(key) + ' from equation constants')
            equation = neweq
            # replace variable in eq with constant
            equation = replaceEqVar(equation ,key,replacedict[key],debug)
    return (equation)


'''
Since Brian2 is only able to build 1D neuron population this script transforms indices to pixel location of the 128x128 DVS and vice versa.
The ind2px function are useful to plot recorded spikes in the same coordinate system to compare the original events as proided by the DVS
'''
def dvs2ind(Events=None, eventDirectory=None, resolution='DAVIS240', scale=True):
    '''
    Input:
        - Events: A numpy array (x, y, ts, pol)
        - eventDirectory: A str containing the path to a .npy file to which holds the 4 x # Events numpy array (x, y, ts, pol)
        - resolution: Specifies the x dimension of the DVS based on the model (e.g. DVS128 or DAVIS240) can be string such as 'DVS128'
          or an integer such as 128
        - scale: A flog to prevent rescaling of timestamps from micro to milliseconds if time stamps are already in milliseconds
    Output:
        - returns a vector of unique indices which maps the pixel location of the camera to the 1D neuron population in brian
    '''
    if eventDirectory is not None:
        assert type(eventDirectory) == str, 'eventDirectory must be a string'
        assert eventDirectory[-4:] == '.npy', 'Please specify a numpy array (.npy) which contains the DVS events.\n Aedat files can be converted using function aedat2numpy.py'
        Events = np.load(eventDirectory)
    if Events is not None:
        assert eventDirectory is None, 'Either you specify a path to load Events using eventDirectory. Or you pass the event numpy directly. NOT Both.'
    if np.size(Events, 0) > np.size(Events, 1):
        Events = np.transpose(Events)

    # extract tempory indices to retrieve 
    cInd_on = Events[3, :] == 1  # Boolean logic to get indices of on and off events, respectively
    cInd_off = Events[3, :] == 0

    # Initialize 1D arrays for neuron indices and timestamps
    indices_on = np.zeros([int(np.sum(cInd_on))])
    spiketimes_on = np.zeros([int(np.sum(cInd_on))])
    # Polarity is either 0 or 1 so the entire length minus the sum of the polarity give the proportion of off events
    indices_off = np.zeros([int(np.sum(cInd_off))])
    spiketimes_off = np.zeros([int(np.sum(cInd_off))])


    if type(resolution) == str:
        resolution = int(resolution[-3:])  # extract the x-resolution (i.e. the resolution along the x-axis of the camera)

    # The equation below follows index = x + y*resolution
    # To retrieve the x and y coordinate again from the index see ind2px
    indices_on = Events[0, cInd_on] + Events[1, cInd_on] * resolution
    indices_off = Events[0, cInd_off] + Events[1, cInd_off] * resolution
    if scale:
        # The DVS timestamps are in microseconds. We need to convert them to milliseconds for brian
        spiketimes_on = np.ceil(Events[2, cInd_on] * 10**(-3))
        spiketimes_off = np.ceil(Events[2, cInd_off] * 10**(-3))
        
    else:
        # The flag scale is used to prevent rescaling of timestamps if we use artifically generated stimuli
        spiketimes_on = np.ceil(Events[2, cInd_on])
        spiketimes_off = np.ceil(Events[2, cInd_off])
        

    
    # Check for double entries within 100 us
    ts_on_tmp = spiketimes_on
    ind_on_tmp = indices_on
    ts_off_tmp = spiketimes_off
    ind_off_tmp = indices_off
    delta_t = 1

    for i in range(len(spiketimes_on)):
        mask_t = spiketimes_on[i]
        mask_i = indices_on[i]

        doubleEntries = np.logical_and(np.logical_and(ts_on_tmp >= mask_t, ts_on_tmp <= mask_t + delta_t), mask_i == ind_on_tmp)
        # uniqueEntries = np.invert(doubleEntries)
        # print np.sum(doubleEntries)
        if np.sum(doubleEntries) > 1:
            tmp = np.where(doubleEntries == True)  # Find first occurence on non-unique entries
            doubleEntries[tmp[0][0]] = False  # keep the first occurance of non-unique entry
            uniqueEntries = np.invert(doubleEntries)
            ts_on_tmp = ts_on_tmp[uniqueEntries]
            ind_on_tmp = ind_on_tmp[uniqueEntries]

    for i in range(len(spiketimes_off)):
        mask_t = spiketimes_off[i]
        mask_i = indices_off[i]

        doubleEntries = np.logical_and(np.logical_and(ts_off_tmp >= mask_t, ts_off_tmp <= mask_t + delta_t), mask_i == ind_off_tmp)
        # uniqueEntries = np.invert(doubleEntries)
        # print np.sum(doubleEntries)
        if np.sum(doubleEntries) > 1:
            tmp = np.where(doubleEntries == True)  # Find first occurence on non-unique entries
            doubleEntries[tmp[0][0]] = False  # keep the first occurance of non-unique entry
            uniqueEntries = np.invert(doubleEntries)
            ts_off_tmp = ts_off_tmp[uniqueEntries]
            ind_off_tmp = ind_off_tmp[uniqueEntries]

    indices_off = ind_off_tmp
    ts_off = ts_off_tmp
    indices_on = ind_on_tmp
    ts_on = ts_on_tmp
    # normalize timestamps
    ts_on -= np.min(ts_on)
    ts_off -= np.min(ts_off)
    return indices_on, ts_on, indices_off, ts_off
