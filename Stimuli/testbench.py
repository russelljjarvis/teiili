'''
This class holds different pre-defined testbench stimuli.
The idea is to test certain aspects of you network with common stimuli.

Author: Moritz Milde
Email: mmilde@ini.uzh.ch
Date: 13.06.2017
'''
from brian2 import *
import numpy as np
import operator


class stdp_testbench():
    def __init__(self, N=1, stimulusLength=1200):
        self.N = N  # Number of Neurons per input group
        self.stimulusLength = stimulusLength

    def stimuli(self):
        '''
        This function returns two brian2 objects.
        Both are Spikegeneratorgroups which hold a single index each
        and varying spike times.
        The protocol follows homoeostasis, weak LTP, weak LTD, strong LTP, strong LTD, homoeostasis
        '''
        t_pre_homoeotasis_1 = np.arange(1, 202, 10)
        t_pre_weakLTP = np.arange(301, 502, 10)
        t_pre_weakLTD = np.arange(601, 802, 10)
        t_pre_strongLTP = np.arange(901, 1102, 10)
        t_pre_strongLTD = np.arange(1201, 1402, 10)
        t_pre_homoeotasis_2 = np.arange(1501, 1702, 10)
        t_pre = np.hstack((t_pre_homoeotasis_1, t_pre_weakLTP, t_pre_weakLTD,
                           t_pre_strongLTP, t_pre_strongLTD, t_pre_homoeotasis_2))

        # Normal distributed shift of spike times to ensure homoeotasis
        t_post_homoeotasis_1 = t_pre_homoeotasis_1 + np.random.rand(len(t_pre_homoeotasis_1))
        t_post_weakLTP = t_pre_weakLTP + 7   # post neuron spikes 7 ms after pre
        t_post_weakLTD = t_pre_weakLTD - 7   # post neuron spikes 7 ms before pre
        t_post_strongLTP = t_pre_strongLTP + 1  # post neurons spikes 1 ms after pre
        t_post_strongLTD = t_pre_strongLTD - 1  # post neurons spikes 1 ms before pre
        t_post_homoeotasis_2 = t_pre_homoeotasis_2 + np.random.randn(len(t_pre_homoeotasis_2))

        t_post = np.hstack((t_post_homoeotasis_1, t_post_weakLTP, t_post_weakLTD,
                            t_post_strongLTP, t_post_strongLTD, t_post_homoeotasis_2))
        ind_pre = np.zeros(len(t_pre))
        ind_post = np.zeros(len(t_post))

        pre = SpikeGeneratorGroup(self.N, indices=ind_pre, times=t_pre * ms, name='gPre')
        post = SpikeGeneratorGroup(self.N, indices=ind_post, times=t_post * ms, name='gPost')
        return pre, post


class octa_testbench():
    def __init__(DVS_SHAPE=(240, 180)):
        self.DVS_SHAPE = DVS_SHAPE
        self.angles = np.arange(-np.pi / 2, np.pi * 3 / 2, 0.01)

    def bar(self):
        '''
        This function returns a single spikegenerator group (brian object)
        The scope of this function is to provide a simple test stimulus
        A bar is moving in 4 directions. The goal is to learn neccessary
        spatiotemporal feature of the mnoving bar and be able to make predictions
        where the bar will move
        '''
        pass

    def infinity(t):
        return np.cos(t), np.sin(t) * np.cos(t)

    def dda_round(x):
        return (x + 0.5).astype(int)

    def translating_bar_infinity(self, length=10, orientation='vertical', shift=32, ts_offset=10, artifical_stimulus=True):
        if not artifical_stimulus:
            if orthogonal == 0:
                fname = 'rec/Inifity_aligned_bar.aedat'
            elif orthogonal == 1:
                fname = 'rec/Infinity_orthogonal_bar.aedat'
            elif orthogonal == 2:
                fname = 'rec/Infinity_orthogonal_aligned_bar.aedat'
            events = aedat2numpy(datafile=fname, camera='DVS240')
            return events
        else:
            x_coord = []
            y_coord = []
            pol = []
            ts = []
            for cAngle in self.angles:
                x, y = self.infinity(cAngle)
                ax.set_xlim((-1.5, 1.5))
                ax.set_ylim((-1.5, 1.5))
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
                self.start = (endx_1, endy_1)
                self.end = (endx_2, endy_2)
                self.max_direction, self.max_length = max(enumerate(abs(self.end - self.start)), key=operator.itemgetter(1))
                self.dv = (self.end - self.start) / self.max_length
                self.line = [dda_round(self.start)]
                for step in range(int(self.max_length)):
                    self.line.append(dda_round((step + 1) * self.dv + self.start))
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
            return events

    def rotating_bar_infinity(self, length=10, orthogonal=False, shift=32, ts_offset=10, artifical_stimulus=True):
        if not artifical_stimulus:
            if orthogonal == 0:
                fname = 'rec/Inifity_aligned_bar.aedat'
            elif orthogonal == 1:
                fname = 'rec/Infinity_orthogonal_bar.aedat'
            elif orthogonal == 2:
                fname = 'rec/Infinity_orthogonal_aligned_bar.aedat'
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

                self.start = (endx_1, endy_1)
                self.end = (endx_2, endy_2)
                self.max_direction, self.max_length = max(enumerate(abs(self.end - self.start)), key=operator.itemgetter(1))
                self.dv = (self.end - self.start) / self.max_length
                self.line = [dda_round(self.start)]
                for step in range(int(self.max_length)):
                    self.line.append(dda_round((step + 1) * self.dv + self.start))
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
            return events

    def ball(self):
        '''
        This function loads a simple recording of a ball moving in a small arena.
        The idea is to test the Online Clustering and Prediction module of OCTAPUS
        The aim is to learn spatio-temporal features based on the ball's trajectory
        and learn to predict its movement
        '''
        events = np.load('rec/ball.npy')
        ind_on, ts_on, ind_off, ts_off = dvs2ind(Events=events, resolution=max(self.DVS_SHAPE), scale=True)
        # depending on how long conversion to index takes we might need to savbe this as well
        input_on = SpikeGeneratorGroup(N=self.DVS_SHAPE[0] * self.DVS_SHAPE[1],
                                       indices=ind_on, times=ts_on, name='input_on*')
        input_off = SpikeGeneratorGroup(N=self.DVS_SHAPE[0] * self.DVS_SHAPE[1],
                                        indices=ind_off, times=ts_off, name='input_off*')
        return input_on, input_off
