'''
This class holds different pre-defined testbench stimuli.
The idea is to test certain aspects of you network with common stimuli.

Author: Moritz Milde
Email: mmilde@ini.uzh.ch
Date: 13.06.2017
'''
from brian2 import *
from NCSBrian2Lib.Tools.tools import dvs2ind, xy2ind, aedat2numpy
import numpy as np
import os
import operator


class stdp_testbench():

    """This class provides a stimuli to test your spike-time depenendent plasticity algorithm

    Attributes:
        N (int): Size of the pre and post neuronal population
        stimulusLength (int): Length of stimuli in ms
    """

    def __init__(self, N=1, stimulusLength=1200):
        """Summary

        Args:
            N (int, optional): Size of the pre and post neuronal population
            stimulusLength (int, optional): Length of stimuli in ms
        """
        self.N = N  # Number of Neurons per input group
        self.stimulusLength = stimulusLength

    def stimuli(self, isi=10):
        """
        This function returns two brian2 objects.
        Both are Spikegeneratorgroups which hold a single index each
        and varying spike times.
        The protocol follows homoeostasis, weak LTP, weak LTD, strong LTP, strong LTD, homoeostasis

        Args:
            isi (int, optional): Interspike Inerval. How many spikes per stimulus phase

        Returns:
            SpikeGeneratorGroup (brian2.obj: rian2 objects which holds the spiketimes as well
                as the respective neuron indices
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
        t_post_homoeotasis_1 = t_pre_homoeotasis_1 + np.clip(np.random.randn(len(t_pre_homoeotasis_1)), -1, 1)
        t_post_weakLTP = t_pre_weakLTP + 5   # post neuron spikes 7 ms after pre
        t_post_weakLTD = t_pre_weakLTD - 5   # post neuron spikes 7 ms before pre
        t_post_strongLTP = t_pre_strongLTP + 1  # post neurons spikes 1 ms after pre
        t_post_strongLTD = t_pre_strongLTD - 1  # post neurons spikes 1 ms before pre
        t_post_homoeotasis_2 = t_pre_homoeotasis_2 + np.clip(np.random.randn(len(t_pre_homoeotasis_2)), -1, 1)

        t_post = np.hstack((t_post_homoeotasis_1, t_post_weakLTP, t_post_weakLTD,
                            t_post_strongLTP, t_post_strongLTD, t_post_homoeotasis_2))
        ind_pre = np.zeros(len(t_pre))
        ind_post = np.zeros(len(t_post))

        pre = SpikeGeneratorGroup(self.N, indices=ind_pre, times=t_pre * ms, name='gPre')
        post = SpikeGeneratorGroup(self.N, indices=ind_post, times=t_post * ms, name='gPost')
        return pre, post


class octa_testbench():

    """This class holds all relevant stimuli to test modules provided with the
    Online Clustering of Temporal Activity (OCTA) framework

    Attributes:
        angles (TYPE): List of angles of orientation
        dv (TYPE): Description
        DVS_SHAPE (TYPE): Input shape of the simulated DVS/DAVIS vision sensor
        end (TYPE): Description
        line (TYPE): Description
        start (TYPE): Description
    """

    def __init__(self, DVS_SHAPE=(240, 180)):
        """Summary

        Args:
            DVS_SHAPE (tuple, optional): Dimension of pixel array of the simulated DVS/DAVIS vision sensor
        """
        self.DVS_SHAPE = DVS_SHAPE
        self.angles = np.arange(-np.pi / 2, np.pi * 3 / 2, 0.01)

    def aedat2events(self, rec, camera='DVS128'):
        """Wrapper function of the original aedat2numpy function in NCSBrian2Lib.Tools.tools.
        This funcion will save events for later usage and will directly return them if no
        SpikeGeneratorGroup is needed.

        Args:
            rec (str): Path to stored .aedat file
            camera (str, optional): Can either be string ('DAVIS240') or int 240, which specifies
                the larger of the 2 pixel dimension to unravel the coordinates into indices

        Returns:
            events (np.ndarray): 4D numpy array with #events entries. Array is organized as x, y, ts, pol
                see aedat2numpy for more details
        """
        assert(type(rec) == str), "rec has to be a string"
        events = aedat2numpy(datafile=rec, camera=camera)
        np.save(rec[:-5] + 'npy', events)
        return events

    def infinity(self, cAngle):
        """Given an angle cAngle this function returns the current postion on an infinity trajectory

        Args:
            cAngle (int): current angle which determines postion on inifinity trajectory

        Returns:
            position (tuple): Postion in x, y coordinates
        """
        return np.cos(cAngle), np.sin(cAngle) * np.cos(cAngle)

    def dda_round(self, x):
        """Simple round funcion

        Args:
            x (float): Value to be rounded

        Returns:
            (int): Ceiled value of x
        """
        return (x + 0.5).astype(int)

    def bar(self, length=10, n2dNeurons=10, orientation='vertical', ts_offset=10,
            angle_step=10, artifical_stimulus=True, rec_path=None):
        """This function returns a single spikegenerator group (Brian object)
        The scope of this function is to provide a simple test stimulus
        A bar is rotating in the center. The goal is to learn necessary
        spatio-temporal feature of the moving bar and be able to make predictions
        where the bar will move

        Args:
            length (int): `length` of the bar in pixel.
            n2dNeurons (int, optional): Size of the pizel array
            orientation (str): `orientation` of the bar. Can either be 'vertical'
                or 'horizontal'
            ts_offset (int): time between two pixel location
            angle_step (int, optional): Angular velocity. Sets setp widh in np.arrange
            artifical_stimulus (bool, optional): Flag if stimulus should be created or loaded from aedat file
            rec_path (str, optional): Path to aedat reacording, only used if arificial_stimulus=False

        Returns:
            SpikeGenerator obj: Brian2 objects which holds the spiketimes as well
                as the respective neuron indices

        Raises:
            UserWarning: If no filename is given but aedat reacording should be loaded
        """
        if not artifical_stimulus:
            if rec_path is None:
                raise UserWarning('No path to recording was provided')
            assert(os.path.isfile(fname)), "No recording exists. Please record the respective stimulus first."
            events = aedat2numpy(datafile=rec_path + 'bar.aedat', camera='DVS240')
        else:
            x_coord = []
            y_coord = []
            pol = []
            ts = []
            center = (n2dNeurons / 2, n2dNeurons / 2)
            self.angles = np.arange(-np.pi / 2, np.pi * 3 / 2, np.radians(angle_step))
            for i, cAngle in enumerate(self.angles):
                endy_1 = center[1] + ((length / 2.) * np.sin((np.pi / 2 + cAngle)))
                endx_1 = center[0] + ((length / 2.) * np.cos((np.pi / 2 + cAngle)))
                endy_2 = center[1] - ((length / 2.) * np.sin((np.pi / 2 + cAngle)))
                endx_2 = center[0] - ((length / 2.) * np.cos((np.pi / 2 + cAngle)))
                self.start = np.asarray((endx_1, endy_1))
                self.end = np.asarray((endx_2, endy_2))
                self.max_direction, self.max_length = max(enumerate(abs(self.end - self.start)),
                                                          key=operator.itemgetter(1))
                self.dv = (self.end - self.start) / self.max_length
                self.line = [self.dda_round(self.start)]
                for step in range(int(self.max_length)):
                    self.line.append(self.dda_round((step + 1) * self.dv + self.start))
                for coord in self.line:
                    x_coord.append(coord[0])
                    y_coord.append(coord[1])
                    ts.append(i * ts_offset)
                    pol.append(1)
            events = np.zeros((4, len(x_coord)))
            events[0, :] = np.asarray(x_coord)
            events[1, :] = np.asarray(y_coord)
            events[2, :] = np.asarray(ts)
            events[3, :] = np.asarray(pol)
        if not artifical_stimulus:
            ind, ts = dvs2ind(events, scale=False)
        else:
            ind = xy2ind(events[0, :], events[1, :], n2dNeurons)
            print(np.max(ind), np.min(ind))
        nPixel = np.int(np.max(ind))
        gInpGroup = SpikeGeneratorGroup(nPixel + 1, indices=ind, times=ts * ms, name='bar')
        return gInpGroup

    def translating_bar_infinity(self, length=10, n2dNeurons=64, orientation='vertical', shift=32,
                                 ts_offset=10, artifical_stimulus=True, rec_path=None,
                                 returnEvents=False):
        """
        This function will either load recorded DAVIS/DVS recordings or generate artificial events
        of bar moving on a infinity trajectory with fixed orientation, i.e. no super-imposed rotation.
        In both cases, the events are provided to a SpikeGeneratorGroup which is returned.

        Args:
            length (int, optional): length of the bar in pixel
            n2dNeurons (int, optional): Size of the pizel array
            orientation (str, optional): lag which determines if bar is orientated vertical or horizontal
            shift (int, optional): offset in x where the stimulus will start
            ts_offset (int, optional): Time in ms between consecutive pixel (stimulus velocity)
            artifical_stimulus (bool, optional): Flag if stimulus should be created or loaded from aedat file
            rec_path (str, optional): Path to recordings
            returnEvents (bool, optional): Flag to return events instead of SpikeGenerator

        Returns:
            SpikeGeneratorGroup (brian2.obj): A SpikeGenerator which has index (i) and spiketimes (t) as attributes
            events (numpy.ndarray, optional): If returnEvents is set events will be returned

        Raises:
            UserWarning: If no filename is given but aedat reacording should be loaded

        """
        if not artifical_stimulus:
            if rec_path is None:
                raise UserWarning('No path to recording was provided')
            if orientation == 'vertical':
                fname = rec_path + 'Inifity_bar_vertical.aedat'
            elif orthogonal == 'horizontal':
                fname = 'Infinity_bar_horizontal.aedat'
            assert(os.path.isfile(fname)), "No recording exists. Please record the respective stimulus first."
            events = aedat2numpy(datafile=fname, camera='DVS240')
        else:
            x_coord = []
            y_coord = []
            pol = []
            ts = []
            for i, cAngle in enumerate(self.angles):
                x, y = self.infinity(cAngle)
                if orientation == 'vertical':
                    endy_1 = y + ((length / 2) * np.sin(np.pi / 2))
                    endx_1 = x + ((length / 2) * np.cos(np.pi / 2))
                    endy_2 = y - ((length / 2) * np.sin(np.pi / 2))
                    endx_2 = x - ((length / 2) * np.cos(np.pi / 2))
                elif orientation == 'horizontal':
                    endy_1 = y + ((length / 2) * np.sin(np.pi))
                    endx_1 = x + ((length / 2) * np.cos(np.pi))
                    endy_2 = y - ((length / 2) * np.sin(np.pi))
                    endx_2 = x - ((length / 2) * np.cos(np.pi))
                self.start = np.asarray((endx_1, endy_1))
                self.end = np.asarray((endx_2, endy_2))
                self.max_direction, self.max_length = max(enumerate(abs(self.end - self.start)),
                                                          key=operator.itemgetter(1))
                self.dv = (self.end - self.start) / self.max_length
                self.line = [self.dda_round(self.start)]
                for step in range(int(self.max_length)):
                    self.line.append(self.dda_round((step + 1) * self.dv + self.start))
                for coord in self.line:
                    x_coord.append(coord[0])
                    y_coord.append(coord[1])
                    ts.append(i * ts_offset)
                    pol.append(1)

            events = np.zeros((4, len(x_coord)))
            events[0, :] = np.asarray(x_coord)
            events[1, :] = np.asarray(y_coord)
            events[2, :] = np.asarray(ts)
            events[3, :] = np.asarray(pol)

        if returnEvents:
            return events
        else:
            if not artifical_stimulus:
                ind, ts = dvs2ind(events, scale=False)
            else:
                ind = xy2ind(events[0, :], events[1, :], n2dNeurons)
                print(np.max(ind), np.min(ind))
            nPixel = np.int(np.max(ind))
            gInpGroup = SpikeGeneratorGroup(nPixel + 1, indices=ind, times=ts * ms, name='bar')
            return gInpGroup

    def rotating_bar_infinity(self, length=10, n2dNeurons=64, orthogonal=False, shift=32,
                              ts_offset=10, artifical_stimulus=True, rec_pah=None,
                              returnEvents=False):
        """This function will either load recorded DAVIS/DVS recordings or generate artificial events
        of bar moving on a infinity trajectory with fixed orientation, i.e. no super-imposed rotation.
        In both cases, the events are provided to a SpikeGeneratorGroup which is returned.

        Args:
            length (int, optional): Length of the bar in pixel
            n2dNeurons (int, optional): Size of the pizel array
            orthogonal (bool, optional): Flag which determines if bar is kept always orthogonal to trajectory,
                if it kept aligned with trajectory or if it returns in "chaotic way"
            shift (int, optional): offset in x where the stimulus will start
            ts_offset (int, optional): Time in ms between consecutive pixel (stimulus velocity)
            artifical_stimulus (bool, optional): Flag if stimulus should be created or loaded from aedat file
            rec_pah (None, optional): Path to recordings
            returnEvents (bool, optional): Flag to return events instead of SpikeGenerator

        Returns:
            SpikeGeneratorGroup (brian2.obj): A SpikeGenerator which has index (i) and spiketimes (t) as attributes
            events (numpy.ndarray, optional): If returnEvents is set events will be returned

        Raises:
            UserWarning: If no filename is given but aedat reacording should be loaded
        """
        if not artifical_stimulus:
            if rec_path is None:
                raise UserWarning('No path to recording was provided')
            if orthogonal == 0:
                fname = 'rec/Inifity_aligned_bar.aedat'
            elif orthogonal == 1:
                fname = 'rec/Infinity_orthogonal_bar.aedat'
            elif orthogonal == 2:
                fname = 'rec/Infinity_orthogonal_aligned_bar.aedat'
            assert(os.path.isfile(fname)), "No recording exists. Please record the respective stimulus first."
            events = aedat2numpy(datafile=fname, camera='DVS240')
            return events
        else:
            x_coord = []
            y_coord = []
            pol = []
            ts = []
            flipped_angles = self.angles[::-1]
            for i, cAngle in enumerate(self.angles):
                x, y = self.infinity(cAngle)
                if orthogonal == 1:
                    if x >= shift:
                        endy_1 = y + ((length / 2) * np.sin((np.pi / 2 * cAngle)))
                        endx_1 = x + ((length / 2) * np.cos((np.pi / 2 * cAngle)))
                        endy_2 = y - ((length / 2) * np.sin((np.pi / 2 * cAngle)))
                        endx_2 = x - ((length / 2) * np.cos((np.pi / 2 * cAngle)))

                    else:
                        endy_1 = y - ((length / 2) * np.sin(np.pi + (np.pi / 2 * flipped_angles[i])))
                        endx_1 = x - ((length / 2) * np.cos(np.pi + (np.pi / 2 * flipped_angles[i])))
                        endy_2 = y + ((length / 2) * np.sin(np.pi + (np.pi / 2 * flipped_angles[i])))
                        endx_2 = x + ((length / 2) * np.cos(np.pi + (np.pi / 2 * flipped_angles[i])))
                elif orthogonal == 0:
                    endy_1 = y + ((length / 2) * np.sin(np.pi / 2 + cAngle))
                    endx_1 = x + ((length / 2) * np.cos(np.pi / 2 + cAngle))
                    endy_2 = y - ((length / 2) * np.sin(np.pi / 2 + cAngle))
                    endx_2 = x - ((length / 2) * np.cos(np.pi / 2 + cAngle))

                elif orthogonal == 2:
                    endy_1 = y + ((length / 2) * np.sin((np.pi / 2 * cAngle)))
                    endx_1 = x + ((length / 2) * np.cos((np.pi / 2 * cAngle)))
                    endy_2 = y - ((length / 2) * np.sin((np.pi / 2 * cAngle)))
                    endx_2 = x - ((length / 2) * np.cos((np.pi / 2 * cAngle)))

                self.start = np.asarray((endx_1, endy_1))
                self.end = np.asarray((endx_2, endy_2))
                self.max_direction, self.max_length = max(enumerate(abs(self.end - self.start)), key=operator.itemgetter(1))
                self.dv = (self.end - self.start) / self.max_length
                self.line = [self.dda_round(self.start)]
                for step in range(int(self.max_length)):
                    self.line.append(self.dda_round((step + 1) * self.dv + self.start))
                for coord in self.line:
                    x_coord.append(coord[0])
                    y_coord.append(coord[1])
                    ts.append(i * ts_offset)
                    pol.append(1)

            events = np.zeros((4, len(x_coord)))
            events[0, :] = np.asarray(x_coord)
            events[1, :] = np.asarray(y_coord)
            events[2, :] = np.asarray(ts)
            events[3, :] = np.asarray(pol)

        if returnEvents:
            return events
        else:
            if not artifical_stimulus:
                ind, ts = dvs2ind(events, scale=False)
            else:
                ind = xy2ind(events[0, :], events[1, :], n2dNeurons)
            nPixel = np.int(np.max(ind))
            gInpGroup = SpikeGeneratorGroup(nPixel + 1, indices=ind, times=ts * ms, name='bar')
            return gInpGroup

    def ball(self, rec_path):
        '''
        This function loads a simple recording of a ball moving in a small arena.
        The idea is to test the Online Clustering and Prediction module of OCTAPUS
        The aim is to learn spatio-temporal features based on the ball's trajectory
        and learn to predict its movement

        Args:
            rec_path (str, required): Path to recording

        Returns:
            SpikeGeneratorGroup (brian2.obj): A SpikeGenerator which has index (i) and spiketimes (t) as attributes

        Raises:
            UserWarning: If no filename is given but aedat reacording should be loaded

        '''
        if rec_path is None:
            raise UserWarning('No path to recording was provided')
        fname = rec_path + 'ball.aedat'
        assert(os.path.isfile(fname)), "No recording ball.aedat exists in {}. Please use jAER to record the stimulus and save it as ball.aedat in {}".format(rec_path, rec_path)
        events = aedat2numpy(datafile=fname, camera='DVS240')
        ind_on, ts_on, ind_off, ts_off = dvs2ind(Events=events, resolution=max(self.DVS_SHAPE), scale=True)
        # depending on how long conversion to index takes we might need to savbe this as well
        input_on = SpikeGeneratorGroup(N=self.DVS_SHAPE[0] * self.DVS_SHAPE[1],
                                       indices=ind_on, times=ts_on, name='input_on*')
        input_off = SpikeGeneratorGroup(N=self.DVS_SHAPE[0] * self.DVS_SHAPE[1],
                                        indices=ind_off, times=ts_off, name='input_off*')
        return input_on, input_off


# class visualize():

#     """Summary

#     Attributes:
#         colors (TYPE): Description
#         eventPlot (TYPE): Description
#         labelStyle (dict): Description
#         win (TYPE): Description
#     """

#     def __init__(self):
#         """Summary
#         """
#         app = QtGui.QApplication([])
#         pg.setConfigOptions(antialias=True)

#         self.colors = [(255, 0, 0), (89, 198, 118), (0, 0, 255), (247, 0, 255),
#                   (0, 0, 0), (255, 128, 0), (120, 120, 120), (0, 171, 255)]
#         self.labelStyle = {'color': '#FFF', 'font-size': '12pt'}
#         self.win = pg.GraphicsWindow(title="Stimuli visualization")
#         self.win.resize(1024, 768)
#         self.eventPlot = self.win.addPlot(title="Input events in ")
#         self.eventPlot.addLegend()

#     def plot(self, events, time_window=35):
#         """Summary

#         Args:
#             events (TYPE): Description
#             time_window (int, optional): Description
#         """
#         if type(events) == str:
#             events = np.load(events)
#         self.eventPlot = self.win.addPlot(title="Input events with size {}x{}".format(np.max(events[0, :]), np.max(events[1, :])))
#         self.eventPlot.addLegend()

#         self.eventPlot.setLabel('left', "Y", **labelStyle)
#         self.eventPlot.setLabel('bottom', "X", **labelStyle)

#         b = QtGui.QFont("Sans Serif", 10)
#         self.eventPlot.getAxis('bottom').tickFont = b
#         self.eventPlot.getAxis('left').tickFont = b

#         # start visualizing
#         start = 0
#         for i in range(0, np.max(events[2, :]), time_window * 10**3):
#             cIndTS = np.logical_and(events[2, :] >= start, events[2, :] <= start + time_window * 10**3)
#             cIndON = np.logical_and(cIndTS, events[3, :] == 1)
#             cIndOFF = np.logical_and(cIndTS, events[3, :] == 0)

#             xData = events[0, cIndON]
#             yData = events[1, cIndON]
#             # plot on events in red
#             self.eventPlot.plot(x=xData, y=yData,
#                  pen=None, symbol='o', symbolPen=None,
#                  symbolSize=7, symbolBrush=self.colors[0],
#                  name='ON Events')

#             if cIndOFF.any():
#                 xData = events[0, cIndOFF]
#                 yData = events[1, cIndOFF]
#                 # plot off events in blue
#                 self.eventPlot.plot(x=xData, y=yData,
#                      pen=None, symbol='o', symbolPen=None,
#                      symbolSize=7, symbolBrush=self.colors[2],
#                      name='OFF Events')

#             start += time_window * 10**3
