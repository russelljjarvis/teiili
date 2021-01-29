from brian2 import ms, TimedArray
import numpy as np
from teili.core.groups import Neurons
from scipy.stats import pearsonr

def neuron_group_from_spikes(spike_indices, spike_times, num_inputs, time_step,
                             duration):
    """Converts spike activity in a neuron group with the same activity.
    
    Args:
        spike_indices (list): Indices of the original source.
        spike_times (list): Time stamps of the original spikes.
        num_inputs (int): Number of input channels from source.
        time_step (brian2.unit.ms): Time step of simulation.
        duration (int): Duration of simulation in samples.

    Returns:
        neu_group (brian2 object): Neuron group with mimicked activity.
    """
    spike_times = [spike_times[np.where(spike_indices==i)[0]]*ms for i in range(num_inputs)]
    # Create matrix where each row (neuron id) is associated with time when there
    # is a spike or -1 when there is not
    converted_input = (np.zeros((num_inputs, duration)) - 1)*ms
    for ind, val in enumerate(spike_times):
        converted_input[ind, np.around(val/time_step).astype(int)] = val
    converted_input = np.transpose(converted_input)
    converted_input = TimedArray(converted_input, dt=time_step)
    # t is simulation time, and will be equal to tspike when there is a spike
    # Cell remains refractory when there is no spike, i.e. tspike=-1
    neu_group = Neurons(num_inputs, model='tspike=converted_input(t, i): second',
            threshold='t==tspike', refractory='tspike < 0*ms')
    neu_group.namespace.update({'converted_input':converted_input})

    return neu_group

def neuron_rate(spike_monitor, kernel_len, kernel_var, kernel_min, interval):
    """Computes firing rates of neurons in a SpikeMonitor.

    Args:
        spike_monitor (brian2.SpikeMonitor): Monitor with spikes and times.
        kernel_len (int): Number of samples in the kernel.
        kernel_var (int): Variance of the kernel window.
        interval (list of int): lower and upper values of the interval, in
            samples, over which the rate will be calculated.

    Returns:
        neuron_rates (dict): Rates and corresponding instants of each neuron.
    """
    spike_trains = spike_monitor.spike_trains()
    neuron_rates = {}
    interval = range(int(interval[0]), int(interval[1])+1)

    # Create normalized and truncated gaussian time window
    kernel_limit = np.floor(kernel_len/2)
    lower_limit = -kernel_limit
    upper_limit = kernel_limit + 1 if kernel_len % 2 else kernel_limit
    kernel = np.exp(-(np.arange(lower_limit, upper_limit)) ** 2 / (2 * kernel_var ** 2))
    kernel = kernel[np.where(kernel>kernel_min)]
    kernel = kernel / kernel.sum()
    for key, neu_spike_times in spike_trains.items():
        # Use histogram to get values that will be convolved
        h, b = np.histogram(neu_spike_times/ms, bins=interval,
                            range=interval)
        neuron_rates[key] = {'rate': np.convolve(h, kernel, mode='same'),
                             't': b[:-1]}

    return neuron_rates

def rate_correlations(rates, interval_dur, intervals):
    """This function uses firing rates of neurons to evaluate the overall
    ensemble formation in a sequence learning task. Considering that the
    average activity of neurons in an ensemble is high when the learned
    symbol is presented, this averege can be correlated to responses to
    each sequence to determine how stable this ensemble is.

    Args:
        rates (dict): Firing rates of each neuron.
        interval_dur (int): Duration of intervals over which sequences are
            presented.
        intervals (int): Number of times sequence is presented.

    Returns:
        correlations (list): Spearman correlations over intervals.
    """
    rate_matrix = np.zeros((len(rates.keys()), interval_dur*intervals))
    for neu, rate in rates.items():
        rate_matrix[neu,:] = rate['rate']

    # Calculates mean rate over presentations of each sequence
    avg_rate = np.zeros((len(rates.keys()), interval_dur))
    for neu in rates.keys():
        temp_rates = np.zeros((intervals, interval_dur))
        for i in range(intervals):
            temp_rates[i, :] = rate_matrix[neu, i*interval_dur:(i+1)*interval_dur]
        avg_rate[neu, :] = np.mean(temp_rates, axis=0)

    # Calculate correlation over sequence presentations
    correlations = []
    for neu in rates.keys():
        for i in range(intervals):
            corr, pval = pearsonr(rate_matrix[neu, i*interval_dur:(i+1)*interval_dur],
                                   avg_rate[neu, :])
            correlations.append(corr)

    return correlations

def ensemble_convergence(input_rates, neuron_rates, input_ensembles,
                         interval_dur, intervals):
    ideal_rate = []
    for group in input_ensembles:
        temp_rate = np.zeros((len(group), interval_dur))
        for i in range(group[0], group[1]):
            temp_rate[i, :] = input_rates[i]['rate']
        ideal_rate.append(temp_rate)

    # TODO return correlation
    # TODO corr(ideal_rate[x], rate_neuron[y] for each trial)
    # TODO make average of above over all neurons
    # TODO
    # temp_rate[i, :] = input_rates[i]['rate']
    # IndexError: index 2 is out of bounds for axis 0 with size 2

    return ideal_rate

#TODO def permutation_from_rate()
