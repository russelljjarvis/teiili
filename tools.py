from brian2 import implementation,check_units,ms,exp,mean,diff
from brian2 import *
import numpy as np
import os
import tempfile
import getpass
import scipy as spv
import struct
import pandas as pd

#===============================================================================
# def setParams(neurongroup, params, debug=False):
#     for par in params:
#         if hasattr(neurongroup, par):
#             setattr(neurongroup, par, params[par])
#     if debug:
#         states = neurongroup.get_states()
#         print ('\n')
#         print ('-_-_-_-_-_-_-_', '\n', 'Parameters set')
#===============================================================================

def printStates(briangroup):
    states = briangroup.get_states()
    print ('\n')
    print ('-_-_-_-_-_-_-_')
    print(briangroup.name)
    print('list of states and first value:')
    for key in states.keys():
        if states[key].size > 1:
            print (key, states[key][1])
        else:
            print (key, states[key])
    print ('----------')


def setParams(briangroup, params, ndargs=None, debug=False):
    for par in params:
        if hasattr(briangroup, par):
            if ndargs is not None and par in ndargs:
                if ndargs[par] is None:
                    setattr(briangroup, par, params[par])
                else:
                    print(par, ndargs, ndargs[par])
                    setattr(briangroup, par, ndargs[par])
            else:
                setattr(briangroup, par, params[par])
    if debug:
        # This fails with synapses coming from SpikeGenerator groups, unidentified bug?
        # This does not work in standalone mode as values of state variables cannot be retrieveed before the simulation has been run
        states = briangroup.get_states()
        print ('\n')
        print ('-_-_-_-_-_-_-_', '\n', 'Parameters set')
        print(briangroup.name)
        print('List of first value of each parameter:')
        for key in states.keys():
            if key in params:
                if states[key].size > 1:
                    print (key, states[key][1])
                else:
                    print (key, states[key])
        print ('----------')


# function that calculates 1D index from 2D index
@implementation('numpy', discard_units=True)
@check_units(x=1, y=1, n2dNeurons=1, result=1)
def xy2ind(x, y, n2dNeurons):
    return int(x) + int(y) * n2dNeurons

# function that calculates 2D index from 1D index


@implementation('numpy', discard_units=True)
@check_units(ind=1, n2dNeurons=1, result=1)
def ind2xy(ind, n2dNeurons):
    ret = (np.mod(np.round(ind), n2dNeurons), np.floor_divide(np.round(ind), n2dNeurons))
    return ret
# Example of indices for n2dNeurons=3
#__0_1_2
#0|0 1 2
#1|3 4 5
#2|6 7 8 
#so ind2xy(4,3) --> (1,1)

# function that calculates distance in 2D field from 2 1D indices

#@implementation('numpy', discard_units=True)
@implementation('cpp', '''
    double fdist2d(int i, int j, int n2dNeurons) {
    int ix = i % n2dNeurons;
    int iy = i / n2dNeurons;
    int jx = j % n2dNeurons;
    int jy = j / n2dNeurons;
    return sqrt(pow((ix - jx),2) + pow((iy - jy),2));
    }
     ''')
@check_units(i=1, j=1, n2dNeurons=1, result=1)
def fdist2d(i, j, n2dNeurons):
    # return sqrt((np.mod(i,n2dNeurons)-np.mod(j,n2dNeurons))**2+(np.floor_divide(i,n2dNeurons)-np.floor_divide(j,n2dNeurons))**2)
    # print(i)
    # print(j)
    # print(n2dNeurons)
    (ix, iy) = ind2xy(i, n2dNeurons)
    (jx, jy) = ind2xy(j, n2dNeurons)
    return np.sqrt((ix - jx)**2 + (iy - jy)**2)

# function that calculates 1D "mexican hat" kernel
@implementation('numpy', discard_units=True)
@check_units(i=1, j=1, sigm=1, result=1)
def fkernel1d(i, j, sigm):
    "function that calculates 1D kernel"
    # res = exp(-((i-j)**2)/(2*sigm**2)) # gaussian, not normalized
    x = i - j
    exponent = -(x**2) / (2 * sigm**2)
    res = (1 + 2 * exponent) * exp(exponent)  # mexican hat, not normalized
    return res

# function that calculates 1D gaussian kernel


@implementation('numpy', discard_units=True)
@check_units(i=1, j=1, sigm=1, result=1)
def fkernelgauss1d(i, j, sigm):
    "function that calculates 1D kernel"
    res = exp(-((i - j)**2) / (2 * sigm**2))  # gaussian, not normalized
    return res


# function that calculates 2D kernel
#@implementation('numpy', discard_units=True)
@implementation('cpp', '''
    double fkernel2d(int i, int j, double gsigma, int n2dNeurons) {
    int ix = i % n2dNeurons;
    int iy = i / n2dNeurons;
    int jx = j % n2dNeurons;
    int jy = j / n2dNeurons;
    int x = ix - jx;
    int y = iy - jy;
    double exponent = -(pow(x,2) + pow(y,2)) / (2 * pow(gsigma,2));
    return ((1 + exponent) * exp(exponent));
    }
     ''')
@check_units(i=1, j=1, gsigma=1, n2dNeurons=1, result=1)
def fkernel2d(i, j, gsigma, n2dNeurons):
    "function that calculates 2D kernel"
    # exponent = -(fdist(i,j,n2dNeurons)**2)/(2*gsigma**2) #alternative
    (ix, iy) = ind2xy(i, n2dNeurons)
    (jx, jy) = ind2xy(j, n2dNeurons)
    x = ix - jx
    y = iy - jy
    exponent = -(x**2 + y**2) / (2 * gsigma**2)
    res = (1 + exponent) * exp(exponent)  # mexican hat / negative Laplacian of Gaussian #not normalized
    return res


def spikemon2firingRate(spikemon,fromT=0*ms,toT="max"):
    spiketimes = (spikemon.t/ms)
    if len(spiketimes)==0:
        return 0
    if toT == "max":
        toT = max(spikemon.t/ms)
    spiketimes = spiketimes[spiketimes<=toT]
    spiketimes = spiketimes[spiketimes>=fromT/ms]
    spiketimes = spiketimes/1000
    if len(spiketimes)==0:
        return 0
    return(mean(1/diff(spiketimes)))



# from Brian2 Equations class
class GenerateWeightMatrix():
    '''
    This module will provide different types of standard weight matrix for 2 neuron population
    Type of weight matrices which are supported:

    Inputs:
        save_path:          Path where to store the weightMatrix. If not set explicitly matrix stored
                            in your temporary folder depending on you OS
        population1:        Brian2 neurongroup
        population2:        Brian2 neurongroup
        mode:               Specifies from which type of distribution weights are sampled
                            Mode can be set to 'normal', 'exponential', 'int', 'uniform'
                            Default: 'normal'
        mu:                 Mean of distribution needed for 'normal' distributed random weights.
                            Default: 0
        sigma:              Standard deviation of distribution needed for 'normal' distributed random weights.
                            Default: 0.25.
        weightRange:        Maximum and minimum value th weights can get. Used by 'int' and 'uniform' mode
        connectionType:     Specifies which type of connectivity is between two populations desired. \
                            Can be either 'fully' (fully connected) or 'sparse'
        connectivityMatrix: If connectionType is sparse, the actual connectivityMatrix needs to be passed to this function
        save:               Flag whether to save created weightMatrix in predefined save_path (see above).
                            If not set weightMatrix will be returned


    Output: If save is set:
                    Weight matrix will be stored in save_path
            if not:
                    Weight matrix will be returned

    Random weight matrix
    Uniformly distributed
    differentiate between fully connected and sparsely connected populations

    Author: Moritz Milde
    Date: 02.12.2016
    '''

    def __init__(self, save_path=None):
        '''
        Init function of the class
        If save_path is not specified but save flag is set (see below) weight matrix
        will be stored in your temporary directory
        '''
        if save_path is not None:
            self.save_path = save_path
            # check if save_path is already created
            self.save_path = os.path.join(save_path, "brian2", "weightMatrices")
            if not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)
        else:
            self.save_path = tempfile.gettempdir()
        self.weightRange = 70
        self.mu = 0
        self.sigma = 0.25

    def getWeights(self):
        '''
        This function generates random weights between two neuron populations specified outside
        Based on the selected mode the random distribution is differently initialized
        Supported modes:
            - 'normal' normal distributed
            - 'int' randomly distributed integers from -range to range
            - 'exponential' weights are sampled from an exponential distribution
            - 'uniform' uniformly spaced weights between -range and range
        '''
        if self.mode == 'normal':
            weightMatrix = np.random.normal(self.mu, self.sigma, size=self.matrixSize)
        elif self.mode == 'int':
            weightMatrix = np.random.randint(-self.weightRange, self.weightRange, self.matrixSize)
        elif self.mode == 'exponential':
            weightMatrix = np.random.exponential(size=self.matrixSize)
        elif self.mode == 'uniform':
            weightMatrix = np.random.uniform(-self.weightRange, self.weightRange, self.matrixSize)
        else:
            raise Exception("%s mode is not supported! Please add the respective function if needed." % (self.mode))
        return weightMatrix

    def randomWeightMatrix(self, population1, population2, mode='normal', mu=None, sigma=None, weightRange=None,
                           connectionType='fully', connectivityMatrix=None, save=False):
        if mu is not None:
            self.mu = mu
        if sigma is not None:
            self.sigma = sigma
        if weightRange is not None:
            self.weightRange = weightRange
        self.matrixSize = (population1.N, population2.N)
        self.mode = mode

        if connectionType == 'fully':
            weightMatrix = self.getWeights()
        # only if we don't have fc populations we need connectivity matrix
        elif connectionType == 'sparse':
            assert(type(connectivityMatrix) is np.ndarray), 'You want a sparse connectivity pattern,\nplease pass the connection matrix'
            self.matrixSize = np.shape(connectivityMatrix)
            weightMatrix = self.getWeights()

        if save:
            np.save(self.save_path + '/weightMatrix.npy', weightMatrix)
            return weightMatrix
        else:
            return weightMatrix


'''
Since Brian2 is only able to build 1D neuron population this script transforms indices to pixel location of the 128x128 DVS and vice versa.
The ind2px function are useful to plot recorded spikes in the same coordinate system to compare the original events as proided by the DVS
'''


def aedat2numpy(datafile='/tmp/aerout.dat', length=0, version="aedat", debug=0, camera='DVS128'):
    """
    load AER data file and parse these properties of AE events:
    - timestamps (in us),
    - x,y-position [0..127]
    - polarity (0/1)
    @param datafile - path to the file to read
    @param length - how many bytes(B) should be read; default 0=whole file
    @param version - which file format version is used: "aedat" = v2, "dat" = v1 (old)
    @param debug - 0 = silent, 1 (default) = print summary, >=2 = print all debug
    @param camera='DVS128' or 'DAVIS240'
    @return (ts, xpos, ypos, pol) 4-tuple of lists containing data of all events;
    """
    # constants
    # V3 = "aedat3"
    # V2 = "aedat" # current 32bit file format
    # V1 = "dat"  # old format
    EVT_DVS = 0  # DVS event type
    EVT_APS = 1  # APS event

    aeLen = 8  # 1 AE event takes 8 bytes
    readMode = '>II'  # struct.unpack(), 2x ulong, 4B+4B
    td = 0.000001  # timestep is 1us
    if(camera == 'DVS128'):
        xmask = 0x00fe
        xshift = 1
        ymask = 0x7f00
        yshift = 8
        pmask = 0x1
        pshift = 0
    elif(camera == 'DAVIS240'):  # values take from scripts/matlab/getDVS*.m
        xmask = 0x003ff000
        xshift = 12
        ymask = 0x7fc00000
        yshift = 22
        pmask = 0x800
        pshift = 11
        eventtypeshift = 31
    else:
        raise ValueError("Unsupported camera: %s" % (camera))
    if (version == V1):
        print ("using the old .dat format")
        aeLen = 6
        readMode = '>HI'  # ushot, ulong = 2B+4B
    aerdatafh = open(datafile, 'rb')
    k = 0  # line number
    p = 0  # pointer, position on bytes
    statinfo = os.stat(datafile)
    if length == 0:
        length = statinfo.st_size
    # print ("file size", length)
    # header
    lt = aerdatafh.readline()
    while lt and lt[0] == "#":
        p += len(lt)
        k += 1
        lt = aerdatafh.readline()
        if debug >= 2:
            print (str(lt))
        continue
    # variables to parse
    timestamps = []
    xaddr = []
    yaddr = []
    pol = []
    # read data-part of file
    aerdatafh.seek(p)
    s = aerdatafh.read(aeLen)
    p += aeLen
    # print (xmask, xshift, ymask, yshift, pmask, pshift)
    while p < length:
        addr, ts = struct.unpack(readMode, s)
        # parse event type
        if(camera == 'DAVIS240'):
            eventtype = (addr >> eventtypeshift)
        else:  # DVS128
            eventtype = EVT_DVS
        # parse event's data
        if(eventtype == EVT_DVS):  # this is a DVS event
            x_addr = (addr & xmask) >> xshift
            y_addr = (addr & ymask) >> yshift
            a_pol = (addr & pmask) >> pshift
            if debug >= 3:
                print("ts->", ts)  # ok
                print("x-> ", x_addr)
                print("y-> ", y_addr)
                print("pol->", a_pol)
            timestamps.append(ts)
            xaddr.append(x_addr)
            yaddr.append(y_addr)
            pol.append(a_pol)
        aerdatafh.seek(p)
        s = aerdatafh.read(aeLen)
        p += aeLen
    if debug > 0:
        try:
            print ("read %i (~ %.2fM) AE events, duration= %.2fs" % (len(timestamps), len(timestamps) / float(10 ** 6), (timestamps[-1] - timestamps[0]) * td))
            n = 5
            print ("showing first %i:" % (n))
            print ("timestamps: %s \nX-addr: %s\nY-addr: %s\npolarity: %s" % (timestamps[0:n], xaddr[0:n], yaddr[0:n], pol[0:n]))
        except:
            print ("failed to print statistics")
    Events = np.zeros([4, len(timestamps)])
    Events[0, :] = xaddr
    Events[1, :] = yaddr
    Events[2, :] = timestamps
    Events[3, :] = pol
    return Events


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
        assert eventDirectory[
            -4:] == '.npy', 'Please specify a numpy array (.npy) which contains the DVS events.\n Aedat files can be converted using function aedat2numpy.py'
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
    return_on = False
    return_off = False
    # normalize timestamps
    if np.size(ts_on) != 0:
        ts_on -= np.min(ts_on)
        return_on = True
    if np.size(ts_off) != 0:
        ts_off -= np.min(ts_off)
        return_off = True
    if return_on == True and return_off == True:
        return indices_on, ts_on, indices_off, ts_off
    elif return_on == True:
        return indices_on, ts_on
    elif return_off == True:
        return indices_off, ts_off
    
def DVScsv2numpy(datafile = 'tmp/aerout.csv', exp_name = 'Experiment', debug = False):
    
    """
    load AER csv logfile and parse these properties of AE events:
    - timestamps (in us),
    - x,y-position [0..127]
    - polarity (0/1)
    @param datafile - path to the file to read
    @param debug - 0 = silent, 1 (default) = print summary, >=2 = print all debug
    @return (ts, xpos, ypos, pol) 4-tuple of lists containing data of all events;
    """
   
    logfile = datafile

    df = pd.read_csv(logfile, header=0)

    df.dropna(inplace=True)
            # Process timestamps: Start at zero
    df['timestamp'] = df['timestamp'].astype(int)

    # Safe raw input
    df['x_raw'] = df['x']
    df['y_raw'] = df['y']
    x_list = []
    y_list = []
    time_list = []
    pol_list = []
    x_list = df['x_raw']
    y_list = df['y_raw']
    time_list = df['timestamp']
    pol_list = df['pol']
    timestep = time_list[0]

    # Get new coordinates with more useful representation
    #df['x'] = df['y_raw']
    #df['y'] = 128 - df['x_raw']
    #discard every third event
    #new_ind = 0
    #Events = np.zeros([4, len(df['timestamp'])/3])
    Events_x = []
    Events_y = []
    Events_time = []
    Events_pol = []
    counter = 0
    for j in range(len(df['timestamp'])):
        if counter % 3 == 0:
            if (timestep == time_list[j]):
                #Events[0, new_ind] = x_list[j]
                Events_x.append(x_list[j])
                Events_y.append(y_list[j])
                Events_time.append(time_list[j])
                Events_pol.append(pol_list[j])
                #new_ind += 1
                timestep = time_list[j]
            else:
                counter += 1
                timestep = time_list[j]
        elif counter % 3 == 1:
            if (timestep == time_list[j]):
                continue
            else:
                counter += 1
                timestep = time_list[j]
        elif counter % 3 == 2:
            if (timestep == time_list[j]):
                continue
            else:
                counter+= 1
                timestep = time_list[j]
    Events = np.zeros([4, len(Events_time)])
    Events[0, :] = Events_x
    Events[1, :] = Events_y
    Events[2, :] = Events_time
    Events[3, :] = Events_pol
    if debug == True:
        print(Events[0, 0:10])
        print(Events[1, 0:10])
        print(Events[2, 0:10])
        print(Events[3, 0:10])
    return Events

