import numpy as np


class DataController(object):
    """ Parent class of all DataControllers"""

    def _filter_events_for_interval(
            self,
            all_spike_times,
            all_neuron_ids,
            t_start,
            t_end):
        """Function to filter for events which are within a given time interval (t_start, t_end)

        Args:
            all_spike_times (list): list of spike times of events
            all_neuron_ids (list): list of neuron ids of events
            t_start (float): start time of time interval within which events should be considered
            t_end (float): end time of time interval within which events should be considered

        Returns:
            all_spike_times (array): array of all spike times of events within specified time interval
            all_neuron_ids (array): array of all neuron ids of events within specified time interval
        """
        all_spike_times = np.asarray(all_spike_times)
        all_neuron_ids = np.asarray(all_neuron_ids)
        shown_indices = np.where(
            np.logical_and(
                all_spike_times >= t_start,
                all_spike_times <= t_end))[0]
        return (all_spike_times[shown_indices], all_neuron_ids[shown_indices])

    def _filter_events_for_neuron_ids(
            self,
            all_spike_times,
            all_neuron_ids,
            active_neuron_ids):
        """Function to filter for events of a subset of neuron ids
        Args:
            all_spike_times (list): list of spike times of events
            all_neuron_ids (list): list of neuron ids of events
            active_neuron_ids (list): list of neuron ids to filter for (neurons with these ids are returned)

        Returns:
            all_spike_times (array): array of all spike times of events of specified neuron ids
            all_neuron_ids (array): array of all neuron ids of events within specified neuron ids
        """

        shown_indices = [*map(lambda x: x in active_neuron_ids, all_neuron_ids)]
        return (
            np.asarray(all_spike_times)[shown_indices],
            np.asarray(all_neuron_ids)[shown_indices])

    def filter_events(
            self,
            all_spike_times,
            all_neuron_ids,
            interval=None,
            neuron_ids=None):
        """Function to filter for events which are within a given time interval (t_start, t_end)
            and of a subset of neuron ids.
            If interval is set to None, the interval is set to the duration of the entire spike train.
            If neuron_ids is set to None, all neuron ids are considered.

        Args:
            all_spike_times (list): list of spike times of events
            all_neuron_ids (list): list of neuron ids of events
            interval (tuple(float, float)): (t_start, t_end) of time interval within which events should be considered
            active_neuron_ids (list): list of neuron ids to filter for (neurons with these ids are returned)

        Returns:
            all_spike_times (array): array of all spike times of events of specified neuron ids
            all_neuron_ids (array): array of all neuron ids of events within specified neuron ids
        """

        all_spike_times, all_neuron_ids = np.asarray(
            all_spike_times), np.asarray(all_neuron_ids)

        if interval is None and neuron_ids is None:
            return (all_spike_times, all_neuron_ids)

        if interval is not None:
            all_spike_times, all_neuron_ids = self._filter_events_for_interval(
                all_spike_times=all_spike_times, all_neuron_ids=all_neuron_ids, t_start=interval[0], t_end=interval[1])
        if neuron_ids is not None:
            all_spike_times, all_neuron_ids = self._filter_events_for_neuron_ids(
                all_spike_times=all_spike_times, all_neuron_ids=all_neuron_ids, active_neuron_ids=neuron_ids)
        return (all_spike_times, all_neuron_ids)
