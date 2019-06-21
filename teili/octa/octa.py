# -*- coding: utf-8 -*-
# @Author: Moritz Milde
# @Date:   2017-12-11 16:16:03
# @Last Modified by:   Moritz Milde
# @Last Modified time: 2018-06-06 12:47:22

"""Summary
Online Clustering of Temporal Activity (OCTA)
Attributes:
    args (TYPE): Pre-defined arguments
    parser (TYPE): Argument parser to define default parameters such as data directories or recording files
"""

from brian2 import *
import numpy as np
import os
import argparse

from teili.core.groups import Neurons, Connections
from teili.building_blocks.wta import WTA
from teili.tools.converter import dvs2ind, aedat2numpy
from teili.tools.indexing import xy2ind, ind2xy
# This set functions should be ultimatley moved to tools.converter
from interfaces import convert


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data_dir",
                    help="Directory of recordings from CLAUS.", type=str,
                    default='/media/moritz/Ellesmera/Robot_data/')
parser.add_argument("--date",
                    help="Date of the recoring in format dd-mm-yyyy/.", type=str,
                    default='04-05-2017/')
parser.add_argument("-rec", "--recording",
                    help="Recording file. Must be hdf5 format", type=str,
                    default='rec1493908121.hdf5')
parser.add_argument("--debug",
                    help="Boolian flag to turn on test bench stimulus. This a controlled stimumulus with precise pattern & spike timings",
                    type=bool, default=False)
parser.add_argument("--DVS_SHAPE",
                    help="Resolution of Dynamic Vision Sensor", type=tuple,
                    default=(240, 180))

args = parser.parse_args()
# Global variables


class InputModule():

    """This class handles input connectivity to OCTA compression modules.
    Primarly it tiles a given 2D input and assignes ID's to respective input
    indices.

    Attributes:
        dvs_shape (tuple): Shape of the 2D Dynamic Vision Sensor (DVS) array. In case of the first layer
            of an OCTA network the dvs_shape is used instead of input_shape
        input_shape (tuple): Shape of the 2D array compression maps which serve as primary input to the
            current layer
        pad_size_x (int, optional): Determines the size of zero padding in x direction
        pad_size_y (int, optional)): Determines the size of zero padding in y direction
        tile_number_x (int): Given the stride and and padding how many tiles can be fit along the x-axis
        tile_number_y (int): Given the stride and and padding how many tiles can be fit along the y-axis
        total_number_tiles (int): The total number of tiles in a given level in the hierachy
    """

    def __init__(self, dvs_shape=(240, 180)):
        """Initialize default paramters.
        TODO: Check BB to organize in simialr manner

        Args:
            dvs_shape (tuple, optional): hape of the 2D Dynamic Vision Sensor (DVS) array. In case of the
            first layer of an OCTA network the dvs_shape is used instead of input_shape
        """
        self.dvs_shape = dvs_shape

    def generate_tiles(self, tile_size=(10, 10), stride=(1, 1), input_shape=(240, 180)):
        """Given an input shape and input size to each tile this function partions the input space
        in as many tiles as can be fit to the inut shape

        Args:
            tile_size (tuple, optional): Input size for each tile/OCTA module. This size
                is used to partition the input, which is defined by the input_shape
            stride (tuple, optional): Overlap of tiles in x and y direction
            input_shape (tuple, optional): Shape of th input to be tiled
        """
        # TODO: Check if we need padding or not
        self.input_shape = input_shape
        self.inputSizeX, self.inputSizeY = self.input_shape
        self.tile_size_x, self.tile_size_y = tile_size
        self.stride_size_x, self.stride_size_y = stride
        self.pad_size_x = (inputSizeX - self.tile_size_x) % (self.tile_size_x - self.stride_size_x)
        self.pad_size_y = (inputSizeY - self.tile_size_y) % (self.tile_size_y - self.stride_size_y)
        if self.pad_size_x != 0 or self.pad_size_y != 0:
            self.inputSizeX += pad_size_x
            self.inputSizeY += pad_size_y
        self.tile_number_x = np.ceil(((inputSizeX - self.tile_size_x) /
                                    (self.tile_size_x - self.stride_size_x)) + 1)
        self.tile_number_y = np.ceil(((inputSizeY - self.tile_size_y) /
                                    (tself.ileSizeY - self.stride_size_y)) + 1)
        self.total_number_tiles = tile_number_x * tile_number_y

    def generate_tile_connections(self, dvs=False):
        """This function will create a connection between all source neurons belonging to one target ID.
        The target ID represents the ID of each compression map.

        Args:
            dvs (bool, optional): Flag to indicate if source layer is a DVS, i.e. primary visual input

        Returns:
            numpy.ndarray: 2D numpy array which consists of connection pairs.
        """
        source = []
        target = []
        for tile_id in range(self.total_number_tiles):
            tileIndX = tile_id % self.tile_number_x
            tileIndY = np.floor(tile_id / self.tile_number_x)
            for pixelX in range((self.tile_size_x - self.stride_size_x) * tileIndX,
                                ((self.tile_size_x - self.stride_size_x) * tileIndX) + self.tile_size_x):
                for pixelY in range((self.tile_size_y - self.stride_size_y) * tileIndY,
                                    ((self.tile_size_y - self.stride_size_y) * tileIndY) + self.tile_size_y):
                    if dvs:
                        self.input_shape = self.dvs_shape
                    ind = xy2ind(x=pixelX, y=pixelY, n2dNeurons=np.max(self.input_shape))
                    source.append(ind)
                    target.append(tile_id)

        tileConnection = np.zeros((2, len(source)))
        tileConnection[0, :] = np.assaray(source)
        tileConnection[1, :] = np.assaray(target)
        return tileConnection


class PredictionModule():
    '''
    Connectivity scheme is preicise/losely topographic
    PMap is bigger than CMap in order to account for context input
    CD based inhibition increases with hierachical depth
    slow exc. from input to PMap
    fast inh. from PMap tp input, with CD mechanism?!

    ## Prediction population
    # PMSizeX, PMSizeY,

    what about recurrent connection within in PU
     --> I think comp. would be not helpful
     --> Fixed rec. connection might make OCTA intracable
     --> gonna try 30% fixed rec. connection. These connections can be subject to plasticity at some point


    # Do I need STDP & RateDP

    Attributes:
        context_fan_in (TYPE): Number of nearest naighbours which share information
        size_x (TYPE): Size of prediction map along x
        size_y (TYPE): Size of prediction map along y
    '''
    def __init__(self, context_fan_in=1, size=16):
        """Summary

        Args:
            context_fan_in (int, optional): Number of nearest naighbours which share information
            size (int, optional): Size of prediction neuronal population
        """
        self.context_fan_in = context_fan_in
        self.size_x = size
        self.size_y = size
        # For now excitatory connections, encoding for predictions errors
        # will project in an all to all fashion. But weak and fixed!

    def generate_feedback_connections(self, tile_id):
        '''Generate connections between prediction population and input.
        These can either be generative by excitatory connections or predictive by inhibitory
        synaptic connections

        Feedback synapses are STDP based synpases
        STDP will install the sparsity constrain of FB connection (P(phi_i, phi_j))

        Args:
            tile_id (int): Tile that needs to be connected

        Returns:
            numpy.ndarray: 2D numpy array which consists of connection pairs.
        '''
        for id in tile_id:
            pass

        return feedback_connections

    def generate_context_connections(self, prediction_units, layer_size=10, topdown=False):
        """Generate (mostly) lateral connection among predictive octa module

        Args:
            prediction_units (numpy.ndarray): Array with ID
            layer_size (int, optional): Size of the layer to convert and index to x,y coordinates
            topdown (bool, optional): Flag to generate top-down rather than lateral connections

        Returns:
            numpy.ndarray: 2D numpy array which consists of connection pairs.
        """
        # Loop over neighbour IDs
        for id in prediction_units:
            x, y = ind2xy(id, layer_size)
            for x_offset in range(-self.context_fan_in, self.context_fan_in + 1):
                for y_offset in range(-self.context_fan_in, self.context_fan_in + 1):
                    if x_offset == 0 and y_offset == 0:
                        # exclude self connection
                        continue
                    if topdown and np.logical_xor(x_offset, y_offset):
                        cInd = xy2ind(x + x_offset, y + y_offset, layer_size)
                        source.append(cInd)
                        target.append(id)
                    elif not topdown:
                        cInd = xy2ind(x + x_offset, y + y_offset, layer_size)
                        source.append(cInd)
                        target.append(id)
        contextConnections = np.zeros((2, len(source)))
        contextConnections[0, :] = np.asarray(source)
        contextConnections[1, :] = np.asarray(target)
        return context_connections
    '''
    ## Context
    # generate pattern! 8 nearest neighour? in same hierachy level plus 4 from hierachy + 1
    # generate connections based on scheme... All to All between respective groups (indicated by IDs)
    '''

    def generateErrorConnections(self, tile_id):
        '''
        Error connections
        generate feedforward connections from Input to prediction neurons
        These connections are slow excitatory
        Since the error connections are not subject to training but rather
        drive the prediction population to install STDP between compression pop
        and prediction pop
        The feedback connections must be voltage based models to only prevent input
        if input can be predicted

        Args:
            tile_id (TYPE): Description

        Returns:
            TYPE: Description
        '''

        # Input/Compression neurons per tile are connected in an all-to-all fashion

        return errorConnections

    def generate_prediction_connections(self):
        '''Function that provides the synaptic connection to the prediction map populations

        Returns:
            TYPE: Description

        '''
        return predictionConnections


class CompressionModule():
    """Provides all neccassary function contruct compressive, sparse feature maps based on
    WTA building_blocks
    """

    def __init__(self):
        """Summary
        """
        pass

    def compression_map(self, c_map_size=7):
        """Builds compressive, competive neuron population

        Args:
            c_map_size (int, optional): Description

        Returns:
            Set of dict: Compressive WTA population
        """
        # Generate WTA population
        cMap = WTA(dimensions=2)
        # Change input synapses
        return cMap


class AttentionModule():
    """
    This module emulates the diffuse feedback role of L1
    This module installs a coincidence based amplification of a signal
    via long range projections

    NOTE This module will be spported once predictions are learned

    Attributes:
        distance (TYPE): Description
        tau (TYPE): Description
    """
    def __init__(self, distance=5):
        """Summary

        Args:
            distance (int, optional): Description
        """
        self.distance = distance
        self.tau = 35 * ms

    def generateAttentionConnections(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return attentionConnections


if __name__ == '__main__':
    iModule = InputModule()
    cModule = CompressionModule()
    pModule = PredictionModule()

    if not args.debug:
        # Loading recording & converting to brian2 readable format
        data = args.data_dir
        rec_dir = args.date
        rec = args.recording
        fname = data + rec_dir + rec[:-5]
        # Convert hdf5 to numpy
        if os.path.isfile(fname + '.npy'):
            # Recording already converted. Loading from file...
            events = np.load(fname + '.npy')
        # maybe add a cut (max number of events per file?!)
        elif rec[-5:] == '.hdf5':
            events = convert.convert.hdf5(file=data + rec_dir + rec[:-5] + '.npy')
        elif rec[-6:] == '.aedat':
            fname = data + rec_dir + rec
            events = aedat2numpy(datafile=fname, camera='DVS240')
        else:
            raise UserWarning('The specified data type is currently not supported')

        # Convert numpy array to brian2 like input vectors split in on and off channels
        ind_on, ts_on, ind_off, ts_off = dvs2ind(Events=events, resolution=max(args.DVS_SHAPE), scale=True)
        # depending on how long conversion to index takes we might need to savbe this as well
        input_on = SpikeGeneratorGroup(N=args.DVS_SHAPE[0] * args.DVS_SHAPE[1], indices=ind_on,
                                       times=ts_on, name='input_on*')
        input_off = SpikeGeneratorGroup(N=args.DVS_SHAPE[0] * args.DVS_SHAPE[1], indices=ind_off,
                                        times=ts_off, name='input_off*')
    else:
        events = octa_testbench.translating_bar_infinity()
        ind_on, ts_on = dvs2ind(Events=events, resolution=max(args.DVS_SHAPE), scale=True)
        input_on = SpikeGeneratorGroup(N=args.DVS_SHAPE[0] * args.DVS_SHAPE[1], indices=ind_on,
                                       times=ts_on, name='input_on*')

    print('Pre-processing done')
    # Input Spike generator group

    # depending on DVS resolution generate n tiles (10 x 10 pixel)
    iModule.generate_tiles(tile_size=(10, 10), stride=(1, 1), input_shape=(240, 180))
    # generate input connectivity for tiles
    tile_connection = iModule.generate_tile_connections()
    for tile_id in iModule.total_number_tiles:
        hierachyID = 1
        cIndex = tile_connection[1, :] == tile_id
        # n times compression unit (this is 1 level of the hierachy)
        c_unit_size = 7
        c_unit_width = np.floor(c_unit_size / 2)
        groupname = 'cMap_{}_{}'.format(hierachyID, tile_id)  # The DVS is the 0th layer. First indicates # in hierachy
        # Should I install
        cModule.CompressionMap()
        # For every compression Module we generate a prediction module

    # n times prediction unit
    # each 16 x 16

    # specify how many levels (depth + ratio between layers)

    # long range feedback connection which are defuse and non-plastic (TBI)

    # Setup synapses
    # Initialize learning synapses from I to C
