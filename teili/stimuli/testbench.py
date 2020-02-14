# -*- coding: utf-8 -*-
"""This class holds different pre-defined testbench stimuli.

The idea is to test certain aspects of you network with common stimuli.

Example:
    >>> import numpy as np
    >>> from brian2 import us, ms
    >>> from pyqtgraph.Qt import QtCore, QtGui
    >>> import pyqtgraph as pg
    >>> from teili.stimuli.testbench import OCTA_Testbench
    >>> from teili.tools.plotter2d import Plotter2d

    >>> app = QtGui.QApplication.instance()
    >>> if app is None:
            app = QtGui.QApplication(sys.argv)
    >>> else:
            print('QApplication instance already exists: %s' % str(app))

    >>> testbench = OCTA_Testbench()
    >>> testbench.rotating_bar(length=10, nrows=10, direction='ccw', ts_offset=3,
                           angle_step=10, noise_probability=0.2, repetitions=90, debug=False)

    In order to visualize it:

    >>> event_monitor = Plotter2d.loaddvs(testbench.events)
    >>> imv1 = event_monitor.plot3d_on_off(plot_dt=10*ms, filtersize=15*ms)

    >>> win = pg.GraphicsWindow(title="DVS Spikes")
    >>> gridlayout = QtGui.QGridLayout(win)
    >>> gridlayout.addWidget(imv1, 1, 1)
    >>> win.resize(1500, 1000)
    >>> win.setLayout(gridlayout)
    >>> win.show()
    >>> win.setWindowTitle('DVS plot')
    >>> imv1.play(10)

    >>> app.exec_()

Todo:
    * As soon as visualizer class is updated, change imports!
"""
# @Author: Moritz Milde
# @Date:   2017-12-17 13:22:16

from brian2 import SpikeGeneratorGroup, PoissonGroup
from brian2 import ms, Hz
from teili.tools.converter import dvs2ind, aedat2numpy
from teili.tools.indexing import xy2ind, ind2xy
import numpy as np
import os
import sys
import operator


class STDP_Testbench():
    """This class provides a stimulus to test your spike-timing dependent plasticity algorithm.

    Attributes:
        N (int): Size of the pre and post neuronal population.
        stimulus_length (int): Length of stimuli in ms.
    """

    def __init__(self, N=1, stimulus_length=1200):
        """Initializes the testbench class.

        Args:
            N (int, optional): Size of the pre and post neuronal population.
            stimulus_length (int, optional): Length of stimuli in ms.
        """
        self.N = N  # Number of Neurons per input group
        self.stimulus_length = stimulus_length

    def stimuli(self, isi=10):
        """Stimulus gneration for STDP protocols.

        This function returns two brian2 objects.
        Both are Spikegeneratorgroups which hold a single index each
        and varying spike times.
        The protocol follows homoeostasis, weak LTP, weak LTD, strong LTP,
        strong LTD, homoeostasis.

        Args:
            isi (int, optional): Interspike Interval. How many spikes per stimulus phase.

        Returns:
            SpikeGeneratorGroup (brian2.obj: Brian2 objects which hold the spiketimes and
                the respective neuron indices.
        """
        t_pre_homoeotasis_1 = np.arange(1, 202, isi)
        t_pre_weakLTP = np.arange(301, 502, isi)
        t_pre_weakLTD = np.arange(601, 802, isi)
        t_pre_strongLTP = np.arange(901, 1102, isi)
        t_pre_strongLTD = np.arange(1201, 1402, isi)
        t_pre_homoeotasis_2 = np.arange(1501, 1702, isi)
        t_pre = np.hstack((t_pre_homoeotasis_1, t_pre_weakLTP, t_pre_weakLTD,
                           t_pre_strongLTP, t_pre_strongLTD, t_pre_homoeotasis_2))

        # Normal distributed shift of spike times to ensure homoeotasis
        t_post_homoeotasis_1 = t_pre_homoeotasis_1 + \
            np.clip(np.random.randn(len(t_pre_homoeotasis_1)), -1, 1)
        t_post_weakLTP = t_pre_weakLTP + 5   # post neuron spikes 7 ms after pre
        t_post_weakLTD = t_pre_weakLTD - 5   # post neuron spikes 7 ms before pre
        t_post_strongLTP = t_pre_strongLTP + 1  # post neurons spikes 1 ms after pre
        t_post_strongLTD = t_pre_strongLTD - 1  # post neurons spikes 1 ms before pre
        t_post_homoeotasis_2 = t_pre_homoeotasis_2 + \
            np.clip(np.random.randn(len(t_pre_homoeotasis_2)), -1, 1)

        t_post = np.hstack((t_post_homoeotasis_1, t_post_weakLTP, t_post_weakLTD,
                            t_post_strongLTP, t_post_strongLTD, t_post_homoeotasis_2))
        ind_pre = np.zeros(len(t_pre))
        ind_post = np.zeros(len(t_post))

        pre = SpikeGeneratorGroup(
            self.N, indices=ind_pre, times=t_pre * ms, name='gPre')
        post = SpikeGeneratorGroup(
            self.N, indices=ind_post, times=t_post * ms, name='gPost')
        return pre, post


class STDGM_Testbench():
    """This class provides a stimulus to test your
    spike-timing dependent gain modulation algorithm.
    """

    def __init__(self, N=1, stimulus_length=1200):
        """Initializes the testbench class.

        Args:
            N (int, optional): Size of the pre and post neuronal population.
            stimulus_length (int, optional): Length of stimuli in ms.
        """
        self.N = N  # Number of Neurons per input group
        self.stimulus_length = stimulus_length

    def stimuli(self, isi):
        """Stimulus gneration for STDGM protocols.

        This function returns two brian2 objects.
        Both are Spikegeneratorgroups which hold a single index each
        and varying spike times.
        The protocol follows homoeostasis, weak LTP, weak LTD, strong LTP,
        strong LTD, homoeostasis.

        Args:
            isi (int, optional): Interspike Interval. How many spikes per stimulus phase.

        Returns:
            SpikeGeneratorGroup (brian2.obj: Brian2 objects which hold the spiketimes and
                the respective neuron indices.
        """
        t_pre_homoeotasis_1 = np.arange(20, 222, isi)
        t_pre_weakLTP = np.arange(301, 502, isi)
        t_pre_weakLTD = np.arange(601, 802, isi)
        t_pre_strongLTP = np.arange(901, 1102, isi)
        t_pre_strongLTD = np.arange(1201, 1402, isi)
        t_pre_homoeotasis_2 = np.arange(1501, 1702, isi)
        t_pre = np.hstack((t_pre_homoeotasis_1, t_pre_weakLTP, t_pre_weakLTD,
                           t_pre_strongLTP, t_pre_strongLTD, t_pre_homoeotasis_2))

        # Normal distributed shift of spike times to ensure homoeotasis
        t_post_homoeotasis_1 = t_pre_homoeotasis_1 + \
            np.random.randint(-20, 20, len(t_pre_homoeotasis_1))
        t_post_weakLTP = t_pre_weakLTP + 7   # post neuron spikes 7 ms after pre
        t_post_weakLTD = t_pre_weakLTD - 7   # post neuron spikes 7 ms before pre
        t_post_strongLTP = t_pre_strongLTP + 1  # post neurons spikes 1 ms after pre
        t_post_strongLTD = t_pre_strongLTD - 1  # post neurons spikes 1 ms before pre
        t_post_homoeotasis_2 = t_pre_homoeotasis_2 + \
            np.random.randint(-20, 20, len(t_pre_homoeotasis_2))

        t_post = np.hstack((t_post_homoeotasis_1, t_post_weakLTP, t_post_weakLTD,
                            t_post_strongLTP, t_post_strongLTD, t_post_homoeotasis_2))
        ind_pre = np.zeros(len(t_pre))
        ind_post = np.zeros(len(t_post))

        pre = SpikeGeneratorGroup(
            self.N, indices=ind_pre, times=t_pre * ms, name='gPre')
        post = SpikeGeneratorGroup(
            self.N, indices=ind_post, times=t_post * ms, name='gPost')
        return pre, post


class OCTA_Testbench():
    """This class holds all relevant stimuli to test modules provided with the
    Online Clustering of Temporal Activity (OCTA) framework.

    Attributes:
        angles (numpy.ndarray): List of angles of orientation.
        DVS_SHAPE (TYPE): Input shape of the simulated DVS/DAVIS vision sensor.
        end (TYPE): End pixel location of the line.
        events (TYPE): Attribute storing events of testbench stimulus.
        indices (TYPE): Attribute storing neuron index of testbench stimulus.
        line (TYPE): Stimulus of the testbench which is used to either generate an interactive
            plot to record stimulus with a DVS/DAVIS camera or coordinates are used to generate
            a SpikeGenerator.
        start (TYPE): Start pixel location of the line.
        times (list): Attribute storing spike times of testbench stimulus.
    """

    def __init__(self, DVS_SHAPE=(240, 180)):
        """Summary

        Args:
            DVS_SHAPE (tuple, optional): Dimension of pixel array of the simulated DVS/DAVIS vision sensor.
        """
        self.DVS_SHAPE = DVS_SHAPE
        self.angles = np.arange(-np.pi / 2, np.pi * 3 / 2, 0.01)

    def aedat2events(self, rec, camera='DVS128'):
        """Wrapper function of the original aedat2numpy function in teili.tools.converter.

        This function will save events for later usage and will directly return them if no
        SpikeGeneratorGroup is needed.

        Args:
            rec (str): Path to stored .aedat file.
            camera (str, optional): Can either be string ('DAVIS240') or int 240, which specifies
                the larger of the 2 pixel dimension to unravel the coordinates into indices.

        Returns:
            events (np.ndarray): 4D numpy array with #events entries. Array is organized as x, y, ts, pol. See aedat2numpy for more details.
        """
        assert(type(rec) == str), "rec has to be a string."
        assert(os.path.isfile(rec)), "File does not exist."
        events = aedat2numpy(datafile=rec, camera=camera)
        np.save(rec[:-5] + 'npy', events)
        return events

    def infinity(self, cAngle):
        """Given an angle cAngle this function returns the current position on an infinity trajectory.

        Args:
            cAngle (float): current angle in rad which determines position on infinity trajectory.

        Returns:
            position (tuple): Postion in x, y coordinates.
        """
        return np.cos(cAngle), np.sin(cAngle) * np.cos(cAngle)

    def dda_round(self, x):
        """Simple round funcion.

        Args:
            x (float): Value to be rounded.

        Returns:
            (int): Ceiled value of x.
        """
        if type(x) is np.ndarray:
            return (x + 0.5).astype(int)
        else:
            return int(x + 0.5)

    def rotating_bar(self, length=10, nrows=10, ncols=None, direction='ccw', ts_offset=10,
                     angle_step=10, artifical_stimulus=True, rec_path=None, save_path=None,
                     noise_probability=None, repetitions=1, debug=False):
        """This function returns a single SpikeGeneratorGroup (Brian object).

        The purpose of this function is to provide a simple test stimulus.
        A bar is rotating in the center. The goal is to learn necessary
        spatio-temporal features of the moving bar and be able to make predictions
        about where the bar will move.

        Args:
            length (int): Length of the bar in pixel.
            nrows (int, optional): X-Axis size of the pixel array.
            ncols (int, optional): Y-Axis size of the pixel array.
            orientation (str): Orientation of the bar. Can either be 'vertical'
                or 'horizontal'.
            ts_offset (int): time between two pixel location.
            angle_step (int, optional): Angular velocity. Sets step width in np.arrange.
            artifical_stimulus (bool, optional): Flag if stimulus should be created or loaded from aedat file.
            rec_path (str, optional): Path/to/stored/location/of/recorded/stimulus.aedat.
            save_path (str, optional): Path to store generated events.
            noise_probability (float, optional): Probability of noise events between 0 and 1.
            repetitions (int, optional): Number of revolutions of the rotating bar.
            debug (bool, optional): Flag to print more detailed output of testbench.

        Returns:
            SpikeGenerator obj: Brian2 objects which holds the spike times as well
                as the respective neuron indices

        Raises:
            UserWarning: If no filename is given but aedat recording should be loaded
        """
        if ncols is None:
            ncols = nrows

        num_neurons = nrows * ncols

        if not artifical_stimulus:
            if rec_path is None:
                raise UserWarning('No path to recording was provided')
            assert(os.path.isfile(rec_path + 'bar.aedat')
                   ), "No recording exists. Please record a stimulus first."
            self.events = aedat2numpy(
                datafile=rec_path + 'bar.aedat', camera='DVS240')
        else:
            x_coord = []
            y_coord = []
            pol = []
            self.times = []
            repetition_offset = 0
            center = (nrows / 2, ncols / 2)
            self.angles = np.arange(-np.pi / 2, np.pi *
                                    3 / 2, np.radians(angle_step))
            if direction == 'cw':
                self.angles = np.flip(self.angles, axis=0)
            for repetition in range(repetitions):
                if repetition_offset != 0:
                    repetition_offset += ts_offset
                for i, cAngle in enumerate(self.angles):
                    endy_1 = center[1] + ((length / 2.)
                                          * np.sin((np.pi / 2 + cAngle)))
                    endx_1 = center[0] + ((length / 2.)
                                          * np.cos((np.pi / 2 + cAngle)))
                    endy_2 = center[1] - ((length / 2.)
                                          * np.sin((np.pi / 2 + cAngle)))
                    endx_2 = center[0] - ((length / 2.)
                                          * np.cos((np.pi / 2 + cAngle)))
                    self.start = np.asarray((endx_1, endy_1))
                    self.end = np.asarray((endx_2, endy_2))
                    self.max_direction, self.max_length = max(enumerate(abs(self.end - self.start)),
                                                              key=operator.itemgetter(1))
                    dv = (self.end - self.start) / self.max_length
                    self.line = [self.dda_round(self.start)]
                    for step in range(int(self.max_length)):
                        self.line.append(self.dda_round(
                            (step + 1) * dv + self.start))
                    list_of_coord = []
                    for coord in self.line:
                        list_of_coord.append((coord[0], coord[1]))
                    for coord in self.line:
                        if coord[0] >= nrows or coord[1] >= nrows:
                            if debug:
                                print("Coordinate larger than input space. x: {}, y: {}".format(
                                    coord[0], coord[1]))
                            continue
                        x_coord.append(coord[0])
                        y_coord.append(coord[1])
                        self.times.append(repetition_offset + (i * ts_offset))
                        pol.append(1)
                        if noise_probability is not None and noise_probability >= np.random.rand():
                            noise_index = np.random.randint(0, num_neurons)
                            noise_x, noise_y = ind2xy(
                                noise_index, nrows, ncols)
                            if (noise_x, noise_y) not in list_of_coord:
                                # print(noise_x, noise_y)
                                # print(list_of_coord)
                                list_of_coord.append((noise_x, noise_y))
                                x_coord.append(noise_x)
                                y_coord.append(noise_y)
                                self.times.append(
                                    repetition_offset + (i * ts_offset))
                                pol.append(1)
                repetition_offset = np.max(self.times)
            self.events = np.zeros((4, len(x_coord)))
            self.events[0, :] = np.asarray(x_coord)
            self.events[1, :] = np.asarray(y_coord)
            self.events[2, :] = np.asarray(self.times)
            self.events[3, :] = np.asarray(pol)
        if debug:
            print("Max X: {}. Max Y: {}".format(
                np.max(self.events[0, :]), np.max(self.events[1, :])))
            print("Stimulus last from {} ms to {} ms".format(
                np.min(self.events[2, :]), np.max(self.events[2, :])))
        if not artifical_stimulus:
            self.indices, self.times = dvs2ind(self.events, scale=False)
        else:
            self.indices = xy2ind(np.asarray(self.events[0, :], dtype='int'), np.asarray(self.events[
                                  1, :], dtype='int'), nrows, ncols)
            if debug:
                print("Maximum index: {}, minimum index: {}".format(
                    np.max(self.indices), np.min(self.indices)))
        nPixel = np.int(np.max(self.indices))
        gInpGroup = SpikeGeneratorGroup(
            nPixel + 1, indices=self.indices, times=self.times * ms, name='bar')
        return gInpGroup

    def translating_bar_infinity(self, length=10, nrows=64, ncols=None, orientation='vertical', shift=32,
                                 ts_offset=10, artifical_stimulus=True, rec_path=None,
                                 return_events=False):
        """
        This function will either load recorded DAVIS/DVS recordings or generate artificial events
        of a bar moving on an infinity trajectory with fixed orientation, i.e. no super-imposed rotation.
        In both cases, the events are provided to a SpikeGeneratorGroup which is returned.

        Args:
            length (int, optional): length of the bar in pixel.
            nrows (int, optional): X-Axis size of the pixel array.
            ncols (int, optional): Y-Axis size of the pixel array.
            orientation (str, optional): lag which determines if bar is orientated vertically or horizontally.
            shift (int, optional): offset in x where the stimulus will start.
            ts_offset (int, optional): Time in ms between consecutive pixels (stimulus velocity).
            artifical_stimulus (bool, optional): Flag if stimulus should be created or loaded from aedat file.
            rec_path (str, optional): Path/to/stored/location/of/recorded/stimulus.aedat.
            return_events (bool, optional): Flag to return events instead of SpikeGenerator.

        Returns:
            SpikeGeneratorGroup (brian2.obj): A SpikeGenerator which has index (i) and spiketimes (t) as attributes.
            events (numpy.ndarray, optional): If return_events is set, events will be returned.

        Raises:
            UserWarning: If no filename is given but aedat recording should be loaded.

        """
        if ncols is None:
            ncols = nrows

        num_neurons = nrows * ncols
        if not artifical_stimulus:
            if rec_path is None:
                raise UserWarning('No path to recording was provided')
            if orientation == 'vertical':
                fname = rec_path + 'Inifity_bar_vertical.aedat'
            elif orientation == 'horizontal':
                fname = 'Infinity_bar_horizontal.aedat'
            assert(os.path.isfile(
                fname)), "No recording exists. Please record a stimulus first."
            self.events = aedat2numpy(datafile=fname, camera='DVS240')
        else:
            x_coord = []
            y_coord = []
            pol = []
            self.times = []
            for i, cAngle in enumerate(self.angles):
                x, y = self.infinity(cAngle)
                if orientation == 'vertical':
                    endy_1 = shift + shift * y + \
                        ((length / 2) * np.sin(np.pi / 2))
                    endx_1 = shift + shift * x + \
                        ((length / 2) * np.cos(np.pi / 2))
                    endy_2 = shift + shift * y - \
                        ((length / 2) * np.sin(np.pi / 2))
                    endx_2 = shift + shift * x - \
                        ((length / 2) * np.cos(np.pi / 2))
                elif orientation == 'horizontal':
                    endy_1 = shift + shift * y + ((length / 2) * np.sin(np.pi))
                    endx_1 = shift + shift * x + ((length / 2) * np.cos(np.pi))
                    endy_2 = shift + shift * y - ((length / 2) * np.sin(np.pi))
                    endx_2 = shift + shift * x - ((length / 2) * np.cos(np.pi))
                self.start = np.asarray((endx_1, endy_1))
                self.end = np.asarray((endx_2, endy_2))
                self.max_direction, self.max_length = max(enumerate(abs(self.end - self.start)),
                                                          key=operator.itemgetter(1))
                dv = (self.end - self.start) / self.max_length
                self.line = [self.dda_round(self.start)]
                for step in range(int(self.max_length)):
                    self.line.append(self.dda_round(
                        (step + 1) * dv + self.start))
                for coord in self.line:
                    x_coord.append(coord[0])
                    y_coord.append(coord[1])
                    self.times .append(i * ts_offset)
                    pol.append(1)

            self.events = np.zeros((4, len(x_coord)))
            self.events[0, :] = np.asarray(x_coord)
            self.events[1, :] = np.asarray(y_coord)
            self.events[2, :] = np.asarray(self.times)
            self.events[3, :] = np.asarray(pol)

        if return_events:
            return self.events
        else:
            if not artifical_stimulus:
                self.indices, self.times = dvs2ind(self.events, scale=False)
            else:
                self.indices = xy2ind(np.asarray(self.events[0, :], dtype='int'),
                                      np.asarray(
                                          self.events[1, :], dtype='int'),
                                      nrows, ncols)
                print(np.max(self.indices), np.min(self.indices))
            nPixel = np.int(np.max(self.indices))
            gInpGroup = SpikeGeneratorGroup(
                nPixel + 1, indices=self.indices, times=self.times * ms, name='bar')
            return gInpGroup

    def rotating_bar_infinity(self, length=10, nrows=64, ncols=None, orthogonal=False, shift=32,
                              ts_offset=10, artifical_stimulus=True, rec_path=None,
                              return_events=False):
        """This function will either load recorded DAVIS/DVS recordings or generate artificial events
        of a bar moving on an infinity trajectory with fixed orientation, i.e. no super-imposed rotation.
        In both cases, the events are provided to a SpikeGeneratorGroup which is returned.

        Args:
            length (int, optional): Length of the bar in pixel.
            nrows (int, optional): X-Axis size of the pixel array.
            ncols (int, optional): Y-Axis size of the pixel array.
            orthogonal (bool, optional): Flag which determines if bar is kept always orthogonal to trajectory,
                if it kept aligned with the trajectory or if it returns in a "chaotic" way.
            shift (int, optional): Offset in x where the stimulus will start.
            ts_offset (int, optional): Time in ms between consecutive pixels (stimulus velocity).
            artifical_stimulus (bool, optional): Flag if stimulus should be created or loaded from aedat file.
            rec_path (str, optional): Path/to/stored/location/of/recorded/stimulus.aedat.
            return_events (bool, optional): Flag to return events instead of SpikeGenerator.

        Returns:
            SpikeGeneratorGroup (brian2.obj): A SpikeGenerator which has index (i) and spiketimes (t) as attributes.
            events (numpy.ndarray, optional): If return_events is set, events will be returned.

        Raises:
            UserWarning: If no filename is given but aedat recording should be loaded.
        """
        if ncols is None:
            ncols = nrows

        num_neurons = nrows * ncols
        if not artifical_stimulus:
            if rec_path is None:
                raise UserWarning('No path to recording was provided')
            if orthogonal == 0:
                fname = rec_path + 'Inifity_aligned_bar.aedat'
            elif orthogonal == 1:
                fname = rec_path + 'Infinity_orthogonal_bar.aedat'
            elif orthogonal == 2:
                fname = rec_path + 'Infinity_orthogonal_aligned_bar.aedat'
            assert(os.path.isfile(
                fname)), "No recording exists. Please record a stimulus first."
            self.events = aedat2numpy(datafile=fname, camera='DVS240')
            return self.events
        else:
            x_coord = []
            y_coord = []
            pol = []
            self.times = []
            flipped_angles = self.angles[::-1]
            for i, cAngle in enumerate(self.angles):
                x, y = self.infinity(cAngle)
                if orthogonal == 1:
                    if x >= shift:
                        endy_1 = shift + shift * y + \
                            ((length / 2) * np.sin((np.pi / 2 * cAngle)))
                        endx_1 = shift + shift * x + \
                            ((length / 2) * np.cos((np.pi / 2 * cAngle)))
                        endy_2 = shift + shift * y - \
                            ((length / 2) * np.sin((np.pi / 2 * cAngle)))
                        endx_2 = shift + shift * x - \
                            ((length / 2) * np.cos((np.pi / 2 * cAngle)))

                    else:
                        endy_1 = shift + shift * y - \
                            ((length / 2) * np.sin(np.pi +
                                                   (np.pi / 2 * flipped_angles[i])))
                        endx_1 = shift + shift * x - \
                            ((length / 2) * np.cos(np.pi +
                                                   (np.pi / 2 * flipped_angles[i])))
                        endy_2 = shift + shift * y + \
                            ((length / 2) * np.sin(np.pi +
                                                   (np.pi / 2 * flipped_angles[i])))
                        endx_2 = shift + shift * x + \
                            ((length / 2) * np.cos(np.pi +
                                                   (np.pi / 2 * flipped_angles[i])))
                elif orthogonal == 0:
                    endy_1 = shift + shift * y + \
                        ((length / 2) * np.sin(np.pi / 2 + cAngle))
                    endx_1 = shift + shift * x + \
                        ((length / 2) * np.cos(np.pi / 2 + cAngle))
                    endy_2 = shift + shift * y - \
                        ((length / 2) * np.sin(np.pi / 2 + cAngle))
                    endx_2 = shift + shift * x - \
                        ((length / 2) * np.cos(np.pi / 2 + cAngle))

                elif orthogonal == 2:
                    endy_1 = shift + shift * y + \
                        ((length / 2) * np.sin((np.pi / 2 * cAngle)))
                    endx_1 = shift + shift * x + \
                        ((length / 2) * np.cos((np.pi / 2 * cAngle)))
                    endy_2 = shift + shift * y - \
                        ((length / 2) * np.sin((np.pi / 2 * cAngle)))
                    endx_2 = shift + shift * x - \
                        ((length / 2) * np.cos((np.pi / 2 * cAngle)))

                self.start = np.asarray((endx_1, endy_1))
                self.end = np.asarray((endx_2, endy_2))
                self.max_direction, self.max_length = max(
                    enumerate(abs(self.end - self.start)), key=operator.itemgetter(1))
                dv = (self.end - self.start) / self.max_length
                self.line = [self.dda_round(self.start)]
                for step in range(int(self.max_length)):
                    self.line.append(self.dda_round(
                        (step + 1) * dv + self.start))
                for coord in self.line:
                    x_coord.append(coord[0])
                    y_coord.append(coord[1])
                    self.times.append(i * ts_offset)
                    pol.append(1)

            self.events = np.zeros((4, len(x_coord)))
            self.events[0, :] = np.asarray(x_coord)
            self.events[1, :] = np.asarray(y_coord)
            self.events[2, :] = np.asarray(self.times)
            self.events[3, :] = np.asarray(pol)

        if return_events:
            return self.events
        else:
            if not artifical_stimulus:
                self.indices, self.times = dvs2ind(self.events, scale=False)
            else:
                self.indices = xy2ind(np.asarray(self.events[0, :], dtype='int'),
                                      np.asarray(
                                          self.events[1, :], dtype='int'),
                                      nrows, ncols)
            nPixel = np.int(np.max(self.indices))
            gInpGroup = SpikeGeneratorGroup(
                nPixel + 1, indices=self.indices, times=self.times * ms, name='bar')
            return gInpGroup

    def ball(self, rec_path):
        '''
        This function loads a simple recording of a ball moving in a small arena.
        The idea is to test the Online Clustering and Prediction module of OCTAPUS.
        The aim is to learn spatio-temporal features based on the ball's trajectory
        and learn to predict its movement.

        Args:
            rec_path (str, required): Path to recording.

        Returns:
            SpikeGeneratorGroup (brian2.obj): A SpikeGenerator which has index (i) and spiketimes (t) as attributes

        Raises:
            UserWarning: If no filename is given but aedat reacording should be loaded

        '''
        if rec_path is None:
            raise UserWarning('No path to recording was provided')
        fname = rec_path + 'ball.aedat'
        assert(os.path.isfile(fname)), "No recording ball.aedat exists in {}. Please use jAER to record the stimulus and save it as ball.aedat in {}".format(
            rec_path, rec_path)
        events = aedat2numpy(datafile=fname, camera='DVS240')
        ind_on, ts_on, ind_off, ts_off = dvs2ind(
            Events=events, resolution=max(self.DVS_SHAPE), scale=True)
        # depending on how long conversion to index takes we might need to
        # savbe this as well
        input_on = SpikeGeneratorGroup(N=self.DVS_SHAPE[0] * self.DVS_SHAPE[1],
                                       indices=ind_on, times=ts_on, name='input_on*')
        input_off = SpikeGeneratorGroup(N=self.DVS_SHAPE[0] * self.DVS_SHAPE[1],
                                        indices=ind_off, times=ts_off, name='input_off*')
        return input_on, input_off


class WTA_Testbench():

    """Collection of functions to test the computational properties of the WTA building_block.

    Attributes:
        indices (numpy.ndarray): Array with neuron indices.
        noise_input (brian2.PoissonGroup): PoissonGroup which provides noise events.
        times (numpy.ndarray): Array with neuron spike times.
    """

    def __init__(self):
        """Summary
        """
        pass

    def stimuli(self, num_neurons=16, dimensions=2, start_time=10, end_time=500, isi=2):
        """This function provides simple test stimuli to test the selection mechanism
        of a WTA population.

        Args:
            num_neurons (int, optional): 1D size of WTA population.
            dimensions (int, optional): Dimension of WTA. Can either be 1 or 2
            start_time (int, optional): Start time when stimulus should start.
            end_time (int, optional): End time when stimulus should stop.
            isi (int, optional): Inter-spike between spike times.

        Raises:
            NotImplementedError: If dimension is not 1 or 2 this error is raised
        """
        self.times = np.arange(start_time, end_time + 1, isi)
        if dimensions == 1:
            self.indices = np.round(np.linspace(
                0, num_neurons - 1, len(self.times)))
        elif dimensions == 2:
            self.indices = np.round(np.linspace(
                0, num_neurons - 1, len(self.times))) + (num_neurons**2 / 2)
        else:
            raise NotImplementedError("only 1 and 2 d WTA available, sorry")

    def background_noise(self, num_neurons=10, rate=10):
        """Provides background noise as Poisson spike trains

        Args:
            num_neurons (int, optional): 1D size of WTA population.
            rate (int, optional): Spike frequency f Poisson noise process.
        """
        num2d_neurons = num_neurons**2
        self.noise_input = PoissonGroup(num2d_neurons, rate * Hz)
