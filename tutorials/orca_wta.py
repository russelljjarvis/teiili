""" This module can be used to scale-up models published by Wang et al. (2018).
    """

import pickle
import numpy as np

from brian2 import ms, amp, mA, Hz, mV, ohm, defaultclock, ExplicitStateUpdater,\
        PoissonGroup, PoissonInput, SpikeMonitor, StateMonitor, PopulationRateMonitor

from teili.building_blocks.building_block import BuildingBlock
from teili.core.groups import Neurons, Connections
from teili.tools.group_tools import add_group_activity_proxy,\
    add_group_params_re_init, add_group_param_init

from teili.tools.misc import DEFAULT_FUNCTIONS
from SLIF_run_regs import add_alt_activity_proxy
from monitor_params import monitor_params

defaultclock.dt = 1 * ms
stochastic_decay = ExplicitStateUpdater('''x_new = f(x,t)''')

class orcaWTA(BuildingBlock):
    """A WTA with diverse inhibitory population. This could represent a single
       layer in a cortical sheet.

    Attributes:
        _groups (dict): Contains all synapses and neurons of the building
            block. For convenience, keys identifying a neuronal population 'x'
            should be 'x_cells', whereas keys identifying a synapse between
            population 'x' and 'y' should be 'x_y'. At the moment, options
            available are 'pyr', 'pv', 'sst', and 'vip'.
        layer (str): Indicates which cortical layer the building block
            represents
    """

    def __init__(self,
                 layer,
                 name,
                 conn_params,
                 pop_params,
                 verbose=False,
                 monitor=False,
                 noise=False):
        """ Generates building block with specified characteristics and
                elements described by Wang et al. (2018).

        Args:
            num_exc_neurons (int): Size of excitatory population.
            ei_ratio (int): Ratio of excitatory versus inhibitory population,
                that is ei_ratio:1 representing exc:inh
            layer (str): Indicates cortical layer that is supposed
                to be mimicked by WTA network. It can be L23, L4, L5,
                or L6.
            name (str, required): Name of the building_block population
            conn_params (ConnectionDescriptor): Class which holds building_block
                specific parameters. It can be find on tutorials/orca_params.py
            pop_params (PopulationDescriptor): Class which holds building_block
                specific parameters. It can be find on tutorials/orca_params.py
            verbose (bool, optional): Flag to gain additional information
            monitor (bool, optional): Flag to indicate presence of monitors
            noise (bool, optional): Flag to determine if background noise is to
                be added to neurons. This is generated with a poisson process.
        """
        BuildingBlock.__init__(self,
                               name,
                               None,
                               None,
                               None,
                               verbose,
                               monitor)

        self._groups = {}
        self.layer = layer

        add_populations(self._groups,
                        group_name=self.name,
                        pop_params=pop_params,
                        verbose=verbose,
                        noise=noise
                        )
        add_connections([*conn_params.intra_prob],
                        self._groups,
                        group_name=self.name,
                        conn_params=conn_params,
                        verbose=verbose)

    def add_input(self,
                  input_group,
                  source_name,
                  targets,
                  conn_params):
        """ This functions add an input group and connections to the building
            block.

        Args:
            input_group (brian2.NeuronGroup): Input to
                building block.
            source_name (str): Name of the input group to be registered.
            targets (list of str): Name of the postsynaptic groups as
                stored in _groups.
            conn_params (ConnectionDescriptor): Class which holds building_block
                specific parameters. It can be find on tutorials/orca_params.py
        """
        conn_ids = [source_name + '_' + target.split('_')[0] for target in targets]
        add_connections(conn_ids,
                        self._groups,
                        group_name=self.name,
                        conn_params=conn_params,
                        external_input=input_group)

    def create_monitors(self, monitor_params):
        """ This creates monitors according to dictionary provided.
        
        Args:
            monitor_params (dict): Contains information about monitors.
                For convenience, some words MUST be present in the name of the
                monitored group, according to what the user wants to record.
                At the moment, for tutorials/orca_wta.py, these are:
                'spike', 'state' (with 'cells' or 'conn'), and 'rate'.
                This must be followed by 'x_cell' or 'conn_x_y' accordingly
                (check class docstring for naming description).
        """
        for key, val in monitor_params.items():
            if 'spike' in key:
                self.monitors[key] = SpikeMonitor(self._groups[val['group']],
                                                  name=self.name + key)
            elif 'state' in key:
                if 'cells' in key:
                    self.monitors[key] = StateMonitor(self._groups[val['group']],
                                                      variables=val['variables'],
                                                      record=val['record'],
                                                      dt=val['mon_dt'],
                                                      name=self.name + key)
                elif 'conn' in key:
                    self.monitors[key] = StateMonitor(self._groups[val['group']],
                                                      variables=val['variables'],
                                                      record=True,
                                                      dt=val['mon_dt'],
                                                      name=self.name + key)
            elif 'rate' in key:
                self.monitors[key] = PopulationRateMonitor(self._groups[val['group']],
                                                           name=self.name + key)

    def save_data(self, monitor_params, path, block=0):
        """ Saves monitor data to disk.
        
        Args:
            monitor_params (dict): Contains information about monitors.
                For convenience, some words MUST be present in the key of the
                monitored group, according to what the user wants to record.
                At the moment, for tutorials/orca_wta.py, these are:
                'spike', 'state' (with 'cells' or 'conn'), and 'rate'.
                This must be followed by 'x_cell' or 'conn_x_y' accordingly
                (check class docstring for naming description).
            block (int, optional): If saving in batches, save with batch number.
        """
        selected_keys = [x for x in monitor_params.keys() if 'spike' in x]
        # TODO np.savez(outfile, **{x_name: x, y_name: y}
        for key in selected_keys:
            # Concatenate data from inhibitory population
            if 'pv_cells' in key:
                pv_times = np.array(self.monitors[key].t/ms)
                pv_indices = np.array(self.monitors[key].i)
            elif 'sst_cells' in key:
                sst_times = np.array(self.monitors[key].t/ms)
                sst_indices = np.array(self.monitors[key].i)
                sst_indices += self._groups['pv_cells'].N
            elif 'vip_cells' in key:
                vip_times = np.array(self.monitors[key].t/ms)
                vip_indices = np.array(self.monitors[key].i)
                vip_indices += (self._groups['pv_cells'].N + self._groups['sst_cells'].N)
            elif 'pyr_cells' in key:
                pyr_times = np.array(self.monitors[key].t/ms)
                pyr_indices = np.array(self.monitors[key].i)

        inh_spikes_t = np.concatenate((pv_times, sst_times, vip_times))
        inh_spikes_i = np.concatenate((pv_indices, sst_indices, vip_indices))
        sorting_index = np.argsort(inh_spikes_t)
        inh_spikes_t = inh_spikes_t[sorting_index]
        inh_spikes_i = inh_spikes_i[sorting_index]

        np.savez(path + f'rasters_{block}.npz',
                 exc_spikes_t=pyr_times,
                 exc_spikes_i=pyr_indices,
                 inh_spikes_t=inh_spikes_t,
                 inh_spikes_i=inh_spikes_i
                 )

        # If there are only a few samples, smoothing operation can create an array
        # which is incompatible with array with spike times. This is then addressed
        # before saving to disk
        selected_keys = [x for x in monitor_params.keys() if 'rate' in x]
        for key in selected_keys:
            if 'pyr_cells' in key:
                exc_rate_t = np.array(self.monitors[key].t/ms)
                if self.monitors[key].rate:
                    exc_rate = np.array(self.monitors[key].smooth_rate(width=10*ms)/Hz)
                else:
                    exc_rate = np.array(self.monitors[key].rate/Hz)
            if 'pv_cells' in key:
                inh_rate_t = np.array(self.monitors[key].t/ms)
                if self.monitors[key].rate:
                    inh_rate = np.array(self.monitors[key].smooth_rate(width=10*ms)/Hz)
                else:
                    inh_rate = np.array(self.monitors[key].rate/Hz)
        # Sometimes the smoothed rate calculated on last block is not the
        # same size as time array. In this cases, raw rate is considered. This
        # means artifacts at the end of simulation
        if len(exc_rate_t) != len(exc_rate):
            exc_rate = np.array(self.monitors['rate_pyr_cells'].rate/Hz)
            inh_rate = np.array(self.monitors['rate_pv_cells'].rate/Hz)
        selected_keys = [x for x in monitor_params.keys() if 'state' in x]
        for key in selected_keys:
            if 'pyr_cells' in key:
                I = self.monitors[key].I
        np.savez(path + f'traces_{block}.npz',
                 I=I, exc_rate_t=exc_rate_t, exc_rate=exc_rate,
                 inh_rate_t=inh_rate_t, inh_rate=inh_rate,
                 )

        # Save targets of recurrent connections as python object
        recurrent_ids, ff_ids, ffi_ids = [], [], []
        for row in range(self._groups['pyr_cells'].N):
            recurrent_ids.append(list(self._groups['pyr_pyr'].j[row, :]))
        for row in range(self._groups['ff_cells'].N):
            ff_ids.append(list(self._groups['ff_pyr'].j[row, :]))
        for row in range(self._groups['pv_cells'].N):
            ffi_ids.append(list(self._groups['ff_pv'].j[row, :]))
        #selected_keys = [x for x in monitor_params.keys() if 'state' in x]
        #for key in selected_keys:
        np.savez_compressed(path + f'matrices_{block}.npz',
            rf=self.monitors['statemon_conn_ff_pyr'].w_plast.astype(np.uint8),
            rfw=self.monitors['statemon_static_conn_ff_pyr'].weight.astype(np.uint8),
            rfi=self.monitors['statemon_conn_ff_pv'].w_plast.astype(np.uint8),
            rfwi=self.monitors['statemon_static_conn_ff_pv'].weight.astype(np.uint8),
            rec_mem=self.monitors['statemon_conn_pyr_pyr'].w_plast.astype(np.uint8),
            rec_ids=recurrent_ids, ff_ids=ff_ids, ffi_ids=ffi_ids
            )
        pickled_monitor = self.monitors['spikemon_pyr_cells'].get_states()
        with open(path + f'pickled_{block}', 'wb') as f:
            pickle.dump(pickled_monitor, f)

def add_populations(_groups,
                    group_name,
                    pop_params,
                    verbose,
                    noise
                    ):
    """ This functions add populations of the building block.

    Args:
        groups (dict): Keys to all neuron and synapse groups.
        group_name (str, required): Name of the building_block population
        pop_params (dict): Parameters used to build populations.
        verbose (bool, optional): Flag to gain additional information
        noise (bool, optional): Flag to determine if background noise is to
            be added to neurons. This is generated with a poisson process.
    """
    temp_groups = {}
    for pop_id, params in pop_params._groups.items():
        neu_type = pop_params.group_plast[pop_id]
        temp_groups[pop_id] = Neurons(
            params['num_neu'],
            equation_builder=pop_params.models[neu_type](num_inputs=params['num_inputs']),
            method=stochastic_decay,
            name=group_name+pop_id,
            verbose=verbose)
        temp_groups[pop_id].set_params(pop_params._base_vals[pop_id])
    
    if noise:
        pyr_noise_cells = PoissonInput(pyr_cells, 'Vm_noise', 1, 3*Hz, 12*mV)
        pv_noise_cells = PoissonInput(pv_cells, 'Vm_noise', 1, 2*Hz, 12*mV)
        sst_noise_cells = PoissonInput(sst_cells, 'Vm_noise', 1, 2*Hz, 12*mV)
        vip_noise_cells = PoissonInput(vip_cells, 'Vm_noise', 1, 2*Hz, 12*mV)
        temp_groups.update({'pyr_noise_cells': pyr_noise_cells,
                            'pv_noise_cells': pv_noise_cells,
                            'sst_noise_cells': sst_noise_cells,
                            'vip_noise_cells': vip_noise_cells})

    _groups.update(temp_groups)

def add_connections(connection_ids,
                    _groups,
                    group_name,
                    conn_params,
                    external_input=None,
                    verbose=False):
    """ This function adds the connections of the building block.

    Args:
        connection_ids (list of str): Identification of each connection
            to be made. Each source and target groups must be present
            in _groups.
        _groups (dict): Keys to all neuron and synapse groups.
        group_name (str, required): Name of the building_block population
        conn_params (ConnectionDescriptor): Class which holds building_block
            specific parameters. It can be find on tutorials/orca_params.py
        external_input (brian2.NeuronGroup, optional): Contains input group
            in case it comes from outside target group.
        verbose (bool, optional): Flag to gain additional information
    """
    temp_conns = {}
    source_group = _groups
    syn_types = conn_params.intra_plast
    connectivities = conn_params.intra_prob
    if external_input is not None:
        # Only one input is connected at a time, so this can be done once
        temp_name = connection_ids[0].split('_')[0]
        source_group[f'{temp_name}_cells'] = external_input
        syn_types = conn_params.input_plast
        connectivities = conn_params.input_prob
    for conn_id in connection_ids:
        source, target = conn_id.split('_')[0], conn_id.split('_')[1]
        syn_type = syn_types[conn_id]
        connectivity = connectivities[conn_id]

        temp_conns[conn_id] = Connections(
            source_group[source+'_cells'], _groups[target+'_cells'],
            equation_builder=conn_params.models[syn_type](),
            method=stochastic_decay,
            name=group_name+source+'_'+target
            )

        if source==target:
            if syn_type == 'reinit':
                temp_conns[conn_id].connect('i!=j', p=1)
            else:
                temp_conns[conn_id].connect('i!=j', p=connectivity)
        else:
            if syn_type == 'reinit':
                temp_conns[conn_id].connect(p=1)
            else:
                temp_conns[conn_id].connect(p=connectivity)
        temp_conns[conn_id].set_params(conn_params._base_vals[conn_id])

        sample_vars = conn_params._sample[conn_id]
        for sample_var in sample_vars: 
            add_group_param_init([temp_conns[conn_id]],
                                 variable=sample_var['variable'],
                                 dist_param=sample_var['dist_param'],
                                 scale=1,
                                 unit=sample_var['unit'],
                                 distribution='normal',
                                 clip_min=sample_var['clip_min'],
                                 clip_max=sample_var['clip_max'])
            # TODO could be fixed in teili, as __getattr__('delay') fails
            try:
                rounded_vals = temp_conns[conn_id].__getattr__(
                        sample_var['variable'])
            except AttributeError:
                if sample_var['variable'] == 'delay':
                    rounded_vals = temp_conns[conn_id].delay
            rounded_vals = rounded_vals / np.abs(sample_var['unit'])
            rounded_vals = np.around(rounded_vals)
            temp_conns[conn_id].__setattr__(
                sample_var['variable'],
                rounded_vals*np.abs(sample_var['unit']))

        if syn_type == 'adp':
            dummy_unit = 1*mV
            _groups[target+'_cells'].variables.add_array(
                'activity_proxy',
                size=_groups[target+'_cells'].N,
                dimensions=dummy_unit.dim)
            _groups[target+'_cells'].variables.add_array(
                'normalized_activity_proxy',
                size=_groups[target+'_cells'].N)
            activity_proxy_group = [_groups[target+'_cells']]
            add_group_activity_proxy(activity_proxy_group,
                                     buffer_size=400,
                                     decay=150)
            temp_conns[conn_id].variance_th = np.random.uniform(
                low=temp_conns[conn_id].variance_th - 0.1,
                high=temp_conns[conn_id].variance_th + 0.1,
                size=len(temp_conns[conn_id]))

        if syn_type == 'altadp':
            dummy_unit = 1*amp
            _groups[target+'_cells'].variables.add_array('activity_proxy',
                                          size=_groups[target+'_cells'].N,
                                          dimensions=dummy_unit.dim)
            _groups[target+'_cells'].variables.add_array('normalized_activity_proxy',
                                          size=_groups[target+'_cells'].N)
            add_alt_activity_proxy([_groups[target+'_cells']],
                                   buffer_size=400,
                                   decay=150)

        # If you want to have sparsity without structural plasticity,
        # set the desired connection probability only
        if syn_type == 'reinit':
            for neu in range(_groups[target+'_cells'].N):
                ffe_zero_w = np.random.choice(
                    external_input.N,
                    int(external_input.N * conn_params.input_prob[conn_id]),
                    replace=False)
                temp_conns[conn_id].weight[ffe_zero_w, neu] = 0
                temp_conns[conn_id].w_plast[ffe_zero_w, neu] = 0

            # TODO for reinit_var in conn_params._reinit_vars[conn_id]
            add_group_params_re_init(groups=[temp_conns[conn_id]],
                                     variable='w_plast',
                                     re_init_variable='re_init_counter',
                                     re_init_threshold=1,
                                     re_init_dt=conn_params._reinit_vars[conn_id]['re_init_dt'],
                                     dist_param=3,
                                     scale=1,
                                     distribution='gamma',
                                     clip_min=0,
                                     clip_max=15,
                                     variable_type='int',
                                     reference='synapse_counter')
            add_group_params_re_init(groups=[temp_conns[conn_id]],
                                     variable='weight',
                                     re_init_variable='re_init_counter',
                                     re_init_threshold=1,
                                     re_init_dt=conn_params._reinit_vars[conn_id]['re_init_dt'],
                                     distribution='deterministic',
                                     const_value=1,
                                     reference='synapse_counter')
            # TODO error when usinng below. Ditch tausyn reinit?
            #add_group_params_re_init(groups=[temp_conns[conn_id]],
            #                         variable='tausyn',
            #                         re_init_variable='re_init_counter',
            #                         re_init_threshold=1,
            #                         re_init_dt=conn_params._reinit_vars[conn_id]['re_init_dt'],
            #                         dist_param=5.5,
            #                         scale=1,
            #                         distribution='normal',
            #                         clip_min=4,
            #                         clip_max=7,
            #                         variable_type='int',
            #                         unit='ms',
            #                         reference='synapse_counter')

    _groups.update(temp_conns)
