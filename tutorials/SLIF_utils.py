from brian2 import ms, TimedArray, check_units, run, SpikeGeneratorGroup,\
        SpikeMonitor, Function
import numpy as np
from teili.core.groups import Neurons
from scipy.stats import pearsonr, spearmanr
from random import randint
from pathlib import Path
import pickle

def replicate_sequence(num_channels, reference_indices, reference_times,
                       sequence_duration, duration):
    # Replicates sequence throughout simulation
    input_spikes = SpikeGeneratorGroup(num_channels, reference_indices,
                                       reference_times,
                                       period=sequence_duration*ms)
    input_monitor = SpikeMonitor(input_spikes)
    print('Generating input...')
    run(duration*ms)
    spike_indices = np.array(input_monitor.i)
    spike_times = np.array(input_monitor.t/ms)

    return spike_indices, spike_times

def neuron_rate(spike_source, kernel_len, kernel_var, kernel_min, interval):
    """Computes firing rates of neurons in a SpikeMonitor.

    Args:
        spike_source (brian2.SpikeMonitor): Source with spikes and times. It
            can be a monitor or a dictionary with {'i': [i1, i2, ...],
            't': [t1, t2, ...]}
        kernel_len (int): Number of samples in the kernel.
        kernel_var (int): Variance of the kernel window.
        interval (list of int): lower and upper values of the interval, in
            samples, over which the rate will be calculated.

    Returns:
        neuron_rates (dict): Rates and corresponding instants of each neuron.
    """
    if isinstance(spike_source, SpikeMonitor):
        spike_trains = spike_source.spike_trains()
    elif isinstance(spike_source, dict):
        # Convert to monitor so spike_trains() can be called
        num_indices = max(spike_source['i']) + 1
        spike_gen = SpikeGeneratorGroup(num_indices,
                                           spike_source['i'],
                                           spike_source['t'])
        spike_mon = SpikeMonitor(spike_gen)
        run(max(spike_source['t']))
        spike_trains = spike_mon.spike_trains()
    else:
        import sys
        sys.exit()
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
    symbol is presented, this average can be correlated to responses to
    each sequence to determine how stable this ensemble is.

    Args:
        rates (dict): Firing rates of each neuron.
        interval_dur (int): Duration of intervals over which sequences are
            presented.
        intervals (int): Number of times sequence is presented.

    Returns:
        correlations (list): Pearson correlations over intervals.
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
    # Generate ideal case that will be used as a metric
    ideal_rate = []
    for group in input_ensembles:
        temp_rate = np.zeros((group[1]-group[0], interval_dur))
        for j, i in enumerate(range(group[0], group[1])):
            temp_rate[j, :] = input_rates[i]['rate']
        ideal_rate.append(np.mean(temp_rate, axis=0))

    # Work with numpy array for convenience
    temp_neu_rates = np.array([list(x['rate']) for x in neuron_rates.values()])
    convergence_matrix = np.zeros((len(ideal_rate), len(neuron_rates.keys()),
                                   intervals))
    # Calculate correlation for all neurons on each symbol presentation
    for ideal_ind in range(convergence_matrix.shape[0]):
        for neu_ind, neu_rate in enumerate(temp_neu_rates):
            for i in range(intervals):
                corr, _ = spearmanr(ideal_rate[ideal_ind], 
                          neu_rate[i*interval_dur:(i+1)*interval_dur],
                          axis=1)
                convergence_matrix[ideal_ind, neu_ind, i] = corr

    return convergence_matrix

def random_integers(a, b, _vectorization_idx):
    random_samples = []
    for sample in range(len(_vectorization_idx)):
        random_samples.append(randint(a, b))

    return np.array(random_samples)
random_integers = Function(random_integers, arg_units=[1, 1], return_unit=1,
                           stateless=False, auto_vectorise=True)

def permutation_from_rate(neurons_rate, window_duration, simulation_dt):
    """This functions uses the instant of maximum firing rate to extract
    permutation indices that can be used to sort a raster plot so that
    an activity trace (relative to a given task) is observed.

    Args:
        neurons_rate (dict): Dictionary with firing rate values for each
            neuron. Keys must be neuron index and 'rate' or 't'.
        window_duration (int): Duration of the averaging time window, in
            brian2.units.
        simulation_dt (int): Time step of the simulation, in brian2.units.

    Returns:
        permutation_ids (list): Permutation indices.
    """
    num_neu = len(neurons_rate.keys())
    window_samples = np.around(window_duration/simulation_dt).astype(int)

    average_rates = np.zeros((num_neu, window_samples))*np.nan
    trials = int(len(neurons_rate[0]['rate']) / window_samples)
    temp_t = np.array([x for x in range(window_samples)]) # Proxy time reference
    peak_instants = {}

    for key in neurons_rate.keys():
        average_rate_neu = []
        for trial in range(trials):
            average_rate_neu.append(
                neurons_rate[key]['rate'][trial*window_samples:(trial+1)*window_samples]
                )
        average_rates[key, :] = np.mean(average_rate_neu, axis=0)

        # Consider only spiking neurons
        if average_rates[key].any():
            # Get first peak found on rate
            peak_index = [np.argmax(average_rates[key])]
            peak_instants[key] = temp_t[peak_index]

    # Add unresponsive neurons again
    sorted_peaks = dict(sorted(peak_instants.items(), key=lambda x: x[1]))
    permutation_ids = [x[0] for x in sorted_peaks.items()]
    [permutation_ids.append(neu) for neu in range(num_neu) if not neu in permutation_ids]

    return permutation_ids

def load_merge_multiple(path_name, file_name, mode='pickle', allow_pickle=False):
    merged_dict = {}
    if mode == 'pickle':
        for pickled_file in Path(path_name).glob(file_name):
            with open(pickled_file, 'rb') as f:
                for key, val in pickle.load(f).items():
                    try:
                        merged_dict.setdefault(key, []).extend(val)
                    except TypeError:
                        merged_dict.setdefault(key, []).append(val)
    elif mode == 'numpy':
        for saved_file in sorted(Path(path_name).glob(file_name),
                                 key=lambda path: int(path.stem.split('_')[1])):
            data = np.load(saved_file, allow_pickle=allow_pickle)
            for key, val in data.items():
                if key == 'rec_w' or key == 'rec_ids':
                    # These are supposed to be always the same. There could be
                    # an if conditional here but that would slow things down
                    merged_dict[key] = val
                else:
                    try:
                        merged_dict[key] = np.hstack((merged_dict[key], val))
                    except KeyError:
                        merged_dict[key] = val

    return merged_dict
