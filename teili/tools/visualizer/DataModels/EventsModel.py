# SPDX-License-Identifier: MIT
# Copyright (c) 2018 University of Zurich

import numpy as np

from .DataModel import DataModel


class EventsModel(DataModel):
    """ Model to hold data of spike events (neuron ids and spike times)"""

    def __init__(self, neuron_ids=None, spike_times=None):
        """ Setup EventsModel
        Args:
            neuron_ids (list/array): neuron ids which spiked
            spike_times (list/array): time points where neurons spiked
        """

        self.neuron_ids = neuron_ids
        self.spike_times = spike_times

        self.attributes_to_save = ['neuron_ids', 'spike_times']


    @classmethod
    def from_brian_spike_monitor(cls, brian_spike_monitor):
        """ Classmethod to init EventsModel from brian spike monitor
        Args:
            brian_spike_monitor: brian2 spike monitor object
        """

        newEventsModel = cls(
            neuron_ids=np.asarray(
                brian_spike_monitor.i), spike_times=np.asarray(
                brian_spike_monitor.t))

        newEventsModel.attributes_to_save = ['neuron_ids', 'spike_times']

        return newEventsModel
