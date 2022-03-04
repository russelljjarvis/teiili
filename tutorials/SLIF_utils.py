import sys
import warnings

from brian2 import mV, ms, second, Hz, TimedArray, check_units, run,\
         SpikeGeneratorGroup, SpikeMonitor, Function, DEFAULT_FUNCTIONS

from scipy.stats import pearsonr, spearmanr
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

import numpy as np
import pandas as pd
from teili.core.groups import Neurons
from teili.tools.converter import aedat2numpy, dvs2ind
from random import randint
from pathlib import Path
import pickle
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

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

def neuron_rate(spike_source, kernel_len, kernel_var, simulation_dt,
                interval=None, smooth=False, trials=1):
    """Computes firing rates of neurons in a SpikeMonitor.

    Args:
        spike_source (brian2.SpikeMonitor): Source with spikes and times. It
            can be a monitor or a dictionary with {'i': [i1, i2, ...],
            't': [t1, t2, ...]}
        kernel_len (Brian2.unit): Length of the averaging kernel in units of
            time.
        kernel_var (Brian2.unit): Variance of the averaging kernel in units of
            time.
        simulation_dt (Brian2.unit): Time scale of simulation's time step
        interval (list of int): lower and upper values of the interval, in
            Brian2 units of time, over which the rate will be calculated.
            If None, the whole recording provided is used.
        smooth (boolean): Flag to indicate whether rate should be calculated
            with a smoothing gaussian window.
        trials (int): Number of trials over which result will be averaged.

    Returns:
        neuron_rates (dict): Rates (in Hz) and corresponding instants of each
            neuron. Rate values are stored in a matrix with each line
            corresponding to rates of a given neuron over time.
    """
    # Genrate inputs
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

    # Convert objects for convenience
    bin_samples = np.around(kernel_len/simulation_dt).astype(int)
    spike_trains = [np.around(val/simulation_dt).astype(int)
        for val in spike_trains.values()]

    # Defines general intervals and bins
    if interval:
        min_sample = np.around(interval[0]/simulation_dt).astype(int)
        max_sample = np.around(interval[1]/simulation_dt).astype(int)
    else:
        min_sample = np.nanmin([min(x) if x.any() else np.nan for x in spike_trains])
        if min_sample < bin_samples:
            min_sample = 0
        else:
            min_sample -= min_sample%bin_samples
        max_sample = np.nanmax([max(x) if x.any() else np.nan for x in spike_trains])
    intervals = range(int(min_sample), int(max_sample))
    if len(intervals) % trials:
        warnings.warn(f'Trials must divide interval in even parts. Using '
                      f' one trial for now...')
        trials = 1

    # Creates regular bins
    intervals = np.array_split(intervals, trials)
    n_bins = (intervals[0][-1]-intervals[0][0]) // bin_samples
    if not n_bins:
        print('number of bins equals zero. Exiting...')
        return None
    bins_length = [bin_samples for _ in range(n_bins)]
    # Creates last irregular bin and update histogram interals
    last_bin_length = (intervals[0][-1]-intervals[0][0]) % bin_samples
    if last_bin_length:
        bins_length.append(last_bin_length)
        n_bins += 1

    rates = np.zeros((np.shape(spike_trains)[0], n_bins, trials))
    spike_times = np.zeros(n_bins)

    for trial in range(trials):
        # TODO vectorize histogram calculation
        for idx, neu_spike_times in enumerate(spike_trains):
            bins = ([intervals[trial][0]]
                  + [intervals[trial][0]+x for x in np.cumsum(bins_length)])
            # Use histogram to get values that will be convolved
            h, b = np.histogram(neu_spike_times,
                bins=bins,
                range=intervals[trial], density=False)
            rates[idx, :, trial] = h/(bins_length*simulation_dt)
    rates = np.sum(rates, axis=2)/trials
    if trials > 1:
        # Use last trial to make spike times start at 0
        spike_times = np.array(bins[:-1]) - bins[0]
    else:
        spike_times = b[1:]

    # Consider values for each simulation dt
    interp_func = interp1d(spike_times, rates, kind='next')
    spike_times = np.arange(min(spike_times), max(spike_times), 1)

    neuron_rates = {}
    rates = interp_func(spike_times)

    if smooth:
        # Create normalized and truncated gaussian time window
        kernel_var /= simulation_dt
        neuron_rates['smoothed'] = gaussian_filter1d(rates,
                                                     kernel_var,
                                                     output=float)*Hz
        # Alternatively use numpy.convolve with normalized window
        #kernel_limit = np.floor(bin_samples/2)
        #lower_limit = -kernel_limit
        #upper_limit = kernel_limit + 1 if bin_samples % 2 else kernel_limit
        #kernel = np.exp(-(np.arange(lower_limit, upper_limit)) ** 2 / (2 * kernel_var ** 2))
        #kernel = kernel[np.where(kernel>.001)]
        #kernel = kernel / kernel.sum()

        #neuron_rates['smoothed'] = np.convolve(
        #    newrate, kernel, mode='same')
        
    neuron_rates['rate'] = rates*Hz
    neuron_rates['t'] = spike_times*simulation_dt

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

def permutation_from_rate(neurons_rate):
    """This functions uses the instant of maximum firing rate to extract
    permutation indices that can be used to sort a raster plot so that
    an activity trace (relative to a given task) is observed.

    Args:
        neurons_rate (dict): Dictionary with firing rate values for each
            neuron. Keys must be neuron index and 'rate' or 't'. Rate values
            is stored in matrix with each line representing rate of a given
            neuron over time

    Returns:
        permutation_ids (list): Permutation indices.
    """
    num_neu = np.shape(neurons_rate['rate'])[0]
    num_samples = np.shape(neurons_rate['rate'])[1]

    average_rates = np.zeros((num_neu, num_samples))*np.nan
    # Proxy time reference
    temp_t = np.array([x for x in range(num_samples)])
    peak_instants = {}

    for neu_id, neu_act in enumerate(neurons_rate['smoothed']):
        average_rates[neu_id, :] = neu_act

        # Consider only spiking neurons
        if average_rates[neu_id].any():
            # Get first peak found on rate
            peak_index = [np.argmax(average_rates[neu_id])]
            peak_instants[neu_id] = temp_t[peak_index]

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
                if key == 'rec_ids' or key == 'ff_ids' or key == 'ffi_ids':
                    # These connections do not change. There could be
                    # an if conditional here but that would slow things down
                    merged_dict[key] = val
                else:
                    try:
                        merged_dict[key] = np.hstack((merged_dict[key], val))
                    except KeyError:
                        merged_dict[key] = val

    return merged_dict

def bar_from_recording(filename):
    """ Converts aedat output with events and save arrays to disk
    
    Args:
        filename (str): Name of the file containing events, e.g.
            'dvSave-7V_normal_fullrec.aedat4'
    """
    print('converting events...')
    ev_npy = aedat2numpy(filename, version='V4')

    print('getting time and indices...')
    i_on, t_on, i_off, t_off = dvs2ind(ev_npy, resolution=(346, 260))

    print('saving on disk...')
    np.savez(filename+'_events', on_indices=i_on, on_times=t_on, off_indices=i_off, off_times=t_off)

def recorded_bar_testbench(filename, num_samples, repetitions):
    events = np.load(filename)
    ref_input_indices = events['off_indices']
    ref_input_times = events['off_times']*ms
    ref_input_indices -= min(ref_input_indices)

    # Sampling input space
    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.plot(ref_input_times/ms, ref_input_indices, '.')
    rng = np.random.default_rng(12345)
    sampled_indices = rng.choice(np.unique(ref_input_indices), num_samples, replace=False)
    sampled_indices = np.in1d(ref_input_indices, sampled_indices)
    ref_input_times = ref_input_times[sampled_indices]
    ref_input_indices = ref_input_indices[sampled_indices]

    # Adjusting input size and removing gaps
    tmp_ind = np.unique(ref_input_indices)
    indices_mapping = {val: ind for ind, val in enumerate(tmp_ind)}
    ref_input_indices = np.vectorize(indices_mapping.get)(ref_input_indices)
    #plt.figure()
    #plt.plot(ref_input_times/ms, ref_input_indices, '.')
    #plt.figure()
    #a=np.unique(ref_input_indices)
    #plt.plot(np.arange(len(a)), a)
    #plt.show()

    # Repeat presentation
    input_times = ref_input_times
    input_indices = ref_input_indices
    recording_gap = 1*second
    for rep in range(repetitions):
        recording_duration = np.max(input_times)
        temp_times = ref_input_times + recording_duration + recording_gap
        input_times = np.concatenate((input_times, temp_times)) * second
        input_indices = np.concatenate((input_indices, ref_input_indices))

    return input_times, input_indices

def get_metrics(spike_monitor):
    data = pd.DataFrame({'i': np.array(spike_monitor.i),
                         't': np.array(spike_monitor.t/ms)})

    spike_indices, spike_times = data.sort_values(['i', 't']).values.T

    neu_ids, id_slices = np.unique(spike_indices, True)
    t_arrays = np.split(spike_times, id_slices[1:])
    max_id = int(max(neu_ids))

    isi = [np.diff(x) for x in t_arrays]
    cv = [np.std(x)/np.mean(x) if len(x)>1 else np.nan for x in isi]

    return isi, cv

def plot_norm_act(monitor):
    plt.figure()
    plt.plot(monitor.normalized_activity_proxy.T)
    plt.xlabel('time (ms)')
    plt.ylabel('normalized activity value')
    plt.title('Normalized activity of all neurons')
    plt.show()

def run_batches(Net, orca, training_blocks, training_duration,
                simulation_dt, path, monitor_params):
    remainder_time = int(np.around(training_duration/simulation_dt)
                         % training_blocks) * simulation_dt
    for block in range(training_blocks):
        block_duration = int(np.around(training_duration/simulation_dt)
                             / training_blocks) * simulation_dt
        Net.run(block_duration, report='stdout', report_period=100*ms)
        # Free up memory
        orca.save_data(monitor_params, path, block)

        # Reinitialization. Remove first so obj reference is not lost
        Net.remove([x for x in orca.monitors.values()])
        orca.monitors = {}
        orca.create_monitors(monitor_params)
        Net.add([x for x in orca.monitors.values()])

    if remainder_time != 0:
        block += 1
        Net.run(remainder_time, report='stdout', report_period=100*ms)
        orca.save_data(monitor_params, path, block)
        Net.remove([x for x in orca.monitors.values()])
        orca.monitors = {}
        orca.create_monitors(monitor_params)
        Net.add([x for x in orca.monitors.values()])
