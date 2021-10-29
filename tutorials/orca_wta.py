""" This module can be used to scale-up models published by Wang et al. (2018).
    """

import os
import pickle
import numpy as np

from brian2 import ms, mA, Hz, mV, ohm, defaultclock, ExplicitStateUpdater,\
        PoissonGroup, PoissonInput, SpikeMonitor, StateMonitor, PopulationRateMonitor

from teili.building_blocks.building_block import BuildingBlock
from teili.core.groups import Neurons, Connections
from teili.models.neuron_models import QuantStochLIF as static_neuron_model
from teili.models.synapse_models import QuantStochSyn as static_synapse_model
from teili.models.synapse_models import QuantStochSynStdp as stdp_synapse_model
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.tools.group_tools import add_group_activity_proxy,\
    add_group_params_re_init, add_group_param_init

from orca_params import connection_probability, excitatory_neurons,\
    inhibitory_neurons, excitatory_synapse_soma, excitatory_synapse_dend,\
    inhibitory_synapse_soma, inhibitory_synapse_dend, synapse_mean_weight,\
    mismatch_neuron_param, mismatch_synapse_param, mismatch_plastic_param,\
    inhibitory_ratio, num_inputs
from monitor_params import monitor_params

# Load other models
path = os.path.expanduser("/home/pablo/git/teili")
model_path = os.path.join(path, "teili", "models", "equations", "")
adp_synapse_model = SynapseEquationBuilder.import_eq(
        model_path + 'StochSynAdp.py')
altadp_synapse_model = SynapseEquationBuilder.import_eq(
        model_path + 'StochAdpIin.py')
istdp_synapse_model = SynapseEquationBuilder.import_eq(
        model_path + 'StochInhStdp.py')
adapt_neuron_model = NeuronEquationBuilder(base_unit='quantized',
        intrinsic_excitability='threshold_adaptation',
        position='spatial')
std_synapse_model = SynapseEquationBuilder.import_eq(
        model_path + 'StochStdStdp.py')
reinit_synapse_model = SynapseEquationBuilder(base_unit='quantized',
        plasticity='quantized_stochastic_stdp',
        structural_plasticity='stochastic_counter')

defaultclock.dt = 1 * ms
stochastic_decay = ExplicitStateUpdater('''x_new = f(x,t)''')

class ORCA_WTA(BuildingBlock):
    """A WTA with diverse inhibitory population. This could represent a single
       layer in a cortical sheet.

    Attributes:
        _groups (dict): Contains all synapses and neurons of the building
            block. For convenience, keys identifying a neuronal population 'x'
            should be 'x_cells', whereas keys identifying a synapse between
            population 'x' and 'y' should be 'x_y'. At the moment, options
            available are 'pyr', 'pv', 'sst', and 'vip'.
    """

    def __init__(self,
                 num_exc_neurons,
                 ei_ratio,
                 layer,
                 name='orca_wta_',
                 connectivity_params=connection_probability,
                 exc_cells_params=excitatory_neurons,
                 inh_cells_params=inhibitory_neurons,
                 exc_soma_params=excitatory_synapse_soma,
                 exc_dend_params=excitatory_synapse_dend,
                 inh_soma_params=inhibitory_synapse_soma,
                 inh_dend_params=inhibitory_synapse_dend,
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
            connectivity_params (dict): Dictionary which holds connectivity
                parameters
            exc_cells_params (dict): Dictionary which holds parameters of
                excitatory neurons
            inh_cells_params (dict): Dictionary which holds parameters of
                inhibitory neurons
            exc_soma_params (dict): Dictionary which holds parameters
                of excitatory connections to somatic compartments.
            exc_dend_params (dict): Dictionary which holds parameters
                of excitatory connections to dendritic compartments.
            inh_soma_params (dict): Dictionary which holds parameters
                of inhibitory connections to somatic compartments
            inh_dend_params (dict): Dictionary which holds parameters
                of inhibitory connections to dendritic compartments
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
        add_populations(self._groups,
                        group_name=self.name,
                        num_exc_neurons=num_exc_neurons,
                        ei_ratio=ei_ratio,
                        ratio_pv=inhibitory_ratio[layer]['pv'],
                        ratio_sst=inhibitory_ratio[layer]['sst'],
                        ratio_vip=inhibitory_ratio[layer]['vip'],
                        verbose=verbose,
                        exc_cells_params=exc_cells_params,
                        inh_cells_params=inh_cells_params,
                        noise=noise
                        )
        add_connections(self._groups,
                        group_name=self.name,
                        verbose=verbose,
                        connectivity_params=connectivity_params,
                        exc_soma_params=exc_soma_params,
                        exc_dend_params=exc_dend_params,
                        inh_soma_params=inh_soma_params,
                        inh_dend_params=inh_dend_params)

    def add_input(self,
                  input_group,
                  input_name,
                  targets,
                  plasticity,
                  target_type,
                  connectivity_params=connection_probability,
                  exc_params=excitatory_synapse_soma,
                  sparsity=None,
                  re_init_dt=None):
        """ This functions add an input group and connections to the building
            block.

        Args:
            input_group (brian2.NeuronGroup): Input to
                building block.
            input_name (str): Name of the input to be registered.
            targets (list of str): Name of the postsynaptic groups as
                stored in _groups.
            plasticity (str): Type of plasticity. It can be 'reinit',
                'stdp', or 'static'.
            target_type (str): Define if targets are excitatory or inhibitory.
            connectivity_params (dict): Dictionary which holds building_block
                specific parameters
            exc_params (dict): Dictionary which holds parameters
                of excitatory connections to compartments.
            sparsity (float): Average percentage of connections from input
                that will be initially disconnected. Structural plasticity
                will change those connections according to activity.
        """
        temp_conns = {}
        if plasticity == 'static':
            syn_model = static_synapse_model
        elif plasticity == 'stdp':
            syn_model = stdp_synapse_model
        elif plasticity == 'reinit':
            syn_model = reinit_synapse_model
        elif plasticity == 'std':
            syn_model = std_synapse_model

        for target in targets:
            target_name = target.split('_')[0]
            temp_conns[f'{input_name}_{target_name}'] = Connections(
                input_group, self._groups[target],
                equation_builder=syn_model(),
                method=stochastic_decay,
                name=self.name+f'{input_name}_{target_name}_conn')

        # Make connections and set params
        syn_objects = {key: val for key, val in temp_conns.items()}
        for key, val in syn_objects.items():
            val.connect(p=connectivity_params[key])
            val.set_params(exc_params[plasticity])


        w_init_group = list(temp_conns.values())
        if target_type=='inhibitory':
            dist_param = synapse_mean_weight['inp_i']
        else:
            dist_param = synapse_mean_weight['inp_e']

        if plasticity == 'static':
            add_group_param_init(w_init_group,
                                 variable='weight',
                                 dist_param=dist_param,
                                 scale=1,
                                 distribution='normal',
                                 clip_min=1,
                                 clip_max=15)
            for g in w_init_group:
                g.__setattr__('weight', np.array(g.weight).astype(int))
        else:
            add_group_param_init(w_init_group,
                                 variable='w_plast',
                                 dist_param=dist_param,
                                 scale=1,
                                 distribution='normal',
                                 clip_min=1,
                                 clip_max=15)
            for g in w_init_group:
                g.__setattr__('w_plast', np.array(g.w_plast).astype(int))
                g.weight = 1

        generate_mismatch(list(temp_conns.values()),
                          mismatch_synapse_param)
        if plasticity != 'static':
            generate_mismatch(list(temp_conns.values()),
                              mismatch_plastic_param)

        # If you want to have sparsity without structural plasticity,
        # just set the desired connection probability
        if sparsity is not None:
            for key, val in syn_objects.items():
                target = key.split('_')[1] + '_cells'
                for neu in range(self._groups[target].N):
                    ffe_zero_w = np.random.choice(input_group.N,
                                                  int(input_group.N*sparsity),
                                                  replace=False)
                    val.weight[ffe_zero_w, neu] = 0
                    val.w_plast[ffe_zero_w, neu] = 0

                if re_init_dt is not None:
                    add_group_params_re_init(groups=[val],
                                             variable='w_plast',
                                             re_init_variable='re_init_counter',
                                             re_init_threshold=1,
                                             re_init_dt=re_init_dt,
                                             dist_param=3,
                                             scale=1,
                                             distribution='gamma',
                                             clip_min=0,
                                             clip_max=15,
                                             variable_type='int',
                                             reference='synapse_counter')
                    add_group_params_re_init(groups=[val],
                                             variable='weight',
                                             re_init_variable='re_init_counter',
                                             re_init_threshold=1,
                                             re_init_dt=re_init_dt,
                                             distribution='deterministic',
                                             const_value=1,
                                             reference='synapse_counter')
                    add_group_params_re_init(groups=[val],
                                             variable='tausyn',
                                             re_init_variable='re_init_counter',
                                             re_init_threshold=1,
                                             re_init_dt=re_init_dt,
                                             dist_param=5.5,
                                             scale=1,
                                             distribution='normal',
                                             clip_min=4,
                                             clip_max=7,
                                             variable_type='int',
                                             unit='ms',
                                             reference='synapse_counter')

        self._groups.update(temp_conns)

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
                Iin0 = self.monitors[key].Iin0
                Iin1 = self.monitors[key].Iin1
                Iin2 = self.monitors[key].Iin2
                Iin3 = self.monitors[key].Iin3
        np.savez(path + f'traces_{block}.npz',
                 Iin0=Iin0, Iin1=Iin1, Iin2=Iin2, Iin3=Iin3,
                 exc_rate_t=exc_rate_t, exc_rate=exc_rate,
                 inh_rate_t=inh_rate_t, inh_rate=inh_rate,
                 )

        # Save targets of recurrent connections as python object
        recurrent_ids = []
        for row in range(self._groups['pyr_cells'].N):
            recurrent_ids.append(list(self._groups['pyr_pyr'].j[row, :]))
        #selected_keys = [x for x in monitor_params.keys() if 'state' in x]
        #for key in selected_keys:
        np.savez_compressed(path + f'matrices_{block}.npz',
            rf=self.monitors['statemon_conn_ff_pyr'].w_plast.astype(np.uint8),
            rfw=self.monitors['statemon_static_conn_ff_pyr'].weight.astype(np.uint8),
            rfi=self.monitors['statemon_conn_ff_pv'].w_plast.astype(np.uint8),
            rfwi=self.monitors['statemon_static_conn_ff_pv'].weight.astype(np.uint8),
            rec_mem=self.monitors['statemon_conn_pyr_pyr'].w_plast.astype(np.uint8),
            rec_ids=recurrent_ids
            )
        pickled_monitor = self.monitors['spikemon_pyr_cells'].get_states()
        with open(path + f'pickled_{block}', 'wb') as f:
            pickle.dump(pickled_monitor, f)

def add_populations(_groups,
                    group_name,
                    num_exc_neurons,
                    ei_ratio,
                    ratio_pv,
                    ratio_sst,
                    ratio_vip,
                    verbose,
                    exc_cells_params,
                    inh_cells_params,
                    noise
                    ):
    """ This functions add populations of the building block.

    Args:
        _groups (dict): Keys to all neuron and synapse groups.
        group_name (str, required): Name of the building_block population
        num_exc_neurons (int): Size of excitatory population.
        ei_ratio (int): Ratio of excitatory versus inhibitory population,
            that is ei_ratio:1 representing exc:inh
        ratio_pv (float, optional): Fraction of inhibitory neurons that
            are PV cells.
        ratio_sst (float, optional): Fraction of inhibitory neurons that
            are SST cells.
        ratio_vip (float, optional): Fraction of inhibitory neurons that
            are VIP cells.
        verbose (bool, optional): Flag to gain additional information
        exc_cells_params (dict): Dictionary which holds parameters of
            excitatory neurons
        inh_cells_params (dict): Dictionary which holds parameters of
            inhibitory neurons
        noise (bool, optional): Flag to determine if background noise is to
            be added to neurons. This is generated with a poisson process.
    """
    # TODO remove when no longer testing, as well as if's
    i_plast = 'plastic_inh0'
    num_inh = int(num_exc_neurons/ei_ratio)
    #num_inh = int(num_exc_neurons/1.6)
    num_pv = int(num_inh * ratio_pv)
    num_pv = num_pv if num_pv else 1
    num_sst = int(num_inh * ratio_sst)
    num_sst = num_sst if num_sst else 1
    num_vip = int(num_inh * ratio_vip)
    num_vip = num_vip if num_vip else 1

    pyr_cells = Neurons(num_exc_neurons,
                        equation_builder=adapt_neuron_model(num_inputs=num_inputs['pyr']),
                        method=stochastic_decay,
                        name=group_name+'pyr_cells',
                        verbose=verbose)

    if i_plast == 'plastic_inh':
        dummy_unit = 1*mV
        pyr_cells.variables.add_array('activity_proxy',
                                       size=pyr_cells.N,
                                       dimensions=dummy_unit.dim)
        pyr_cells.variables.add_array('normalized_activity_proxy',
                                       size=pyr_cells.N)

    pv_cells = Neurons(num_pv,
                       equation_builder=adapt_neuron_model(num_inputs=num_inputs['pv']),
                       method=stochastic_decay,
                       name=group_name+'pv_cells',
                       verbose=verbose)
    # TODO organize alt adp below
    from brian2 import amp
    dummy_unit = 1*amp
    pv_cells.variables.add_array('activity_proxy',
                                  size=pv_cells.N,
                                  dimensions=dummy_unit.dim)
    pv_cells.variables.add_array('normalized_activity_proxy',
                                  size=pv_cells.N)
    sst_cells = Neurons(num_sst,
                        equation_builder=static_neuron_model(num_inputs=num_inputs['sst']),
                        method=stochastic_decay,
                        name=group_name+'sst_cells',
                        verbose=verbose)
    vip_cells = Neurons(num_vip,
                        equation_builder=static_neuron_model(num_inputs=num_inputs['vip']),
                        method=stochastic_decay,
                        name=group_name+'vip_cells',
                        verbose=verbose)
    if noise:
        pyr_noise_cells = PoissonInput(pyr_cells, 'Vm_noise', 1, 3*Hz, 12*mV)
        pv_noise_cells = PoissonInput(pv_cells, 'Vm_noise', 1, 2*Hz, 12*mV)
        sst_noise_cells = PoissonInput(sst_cells, 'Vm_noise', 1, 2*Hz, 12*mV)
        vip_noise_cells = PoissonInput(vip_cells, 'Vm_noise', 1, 2*Hz, 12*mV)

    pyr_cells.set_params(exc_cells_params)
    pv_cells.set_params(inh_cells_params['pv'])
    sst_cells.set_params(inh_cells_params['sst'])
    vip_cells.set_params(inh_cells_params['vip'])

    generate_mismatch([pyr_cells, pv_cells, sst_cells, vip_cells],
                      mismatch_neuron_param)

    temp_groups = {'pyr_cells': pyr_cells,
                   'pv_cells': pv_cells,
                   'sst_cells': sst_cells,
                   'vip_cells': vip_cells}
    if noise:
        temp_groups.update({'pyr_noise_cells': pyr_noise_cells,
                            'pv_noise_cells': pv_noise_cells,
                            'sst_noise_cells': sst_noise_cells,
                            'vip_noise_cells': vip_noise_cells})
    _groups.update(temp_groups)

def add_connections(_groups,
                    group_name,
                    connectivity_params,
                    exc_soma_params,
                    exc_dend_params,
                    inh_soma_params,
                    inh_dend_params,
                    verbose):
    """ This function adds the connections of the building block.

    Args:
        _groups (dict): Keys to all neuron and synapse groups.
        group_name (str, required): Name of the building_block population
        connectivity_params (dict): Dictionary which holds building_block
            specific parameters
        exc_soma_params (dict): Dictionary which holds parameters
            of excitatory connections to somatic compartments.
        exc_dend_params (dict): Dictionary which holds parameters
            of excitatory connections to dendritic compartments.
        inh_soma_params (dict): Dictionary which holds parameters
            of inhibitory connections to somatic compartments.
        inh_dend_params (dict): Dictionary which holds parameters
            of inhibitory connections to dendritic compartments.
        verbose (bool, optional): Flag to gain additional information
    """
    # TODO remove when no longer testing, as well as if's
    i_plast = 'plastic_inh0'

    # Creating connections
    # From Pyramidal neurons
    pyr_pyr_conn = Connections(_groups['pyr_cells'], _groups['pyr_cells'],
            equation_builder=stdp_synapse_model(),
            method=stochastic_decay,
            name=group_name+'pyr_pyr_conn')
    pyr_pv_conn = Connections(_groups['pyr_cells'], _groups['pv_cells'],
            equation_builder=static_synapse_model(),
            method=stochastic_decay,
            name=group_name+'pyr_pv_conn')
    pyr_sst_conn = Connections(_groups['pyr_cells'], _groups['sst_cells'],
            equation_builder=static_synapse_model(),
            method=stochastic_decay,
            name=group_name+'pyr_sst_conn')
    pyr_vip_conn = Connections(_groups['pyr_cells'], _groups['vip_cells'],
            equation_builder=static_synapse_model(),
            method=stochastic_decay,
            name=group_name+'pyr_vip_conn')

    # Interneurons to pyramidal
    if i_plast == 'plastic_inh':
        pv_pyr_conn = Connections(_groups['pv_cells'], _groups['pyr_cells'],
                equation_builder=adp_synapse_model,
                method=stochastic_decay,
                name=group_name+'pv_pyr_conn')
        sst_pyr_conn = Connections(_groups['sst_cells'], _groups['pyr_cells'],
                                   equation_builder=adp_synapse_model,
                                   method=stochastic_decay,
                                   name=group_name+'sst_pyr_conn')
    elif i_plast == 'plastic_inh0':
        pv_pyr_conn = Connections(_groups['pv_cells'], _groups['pyr_cells'],
                                  equation_builder=istdp_synapse_model,
                                  method=stochastic_decay,
                                  name=group_name+'pv_pyr_conn')
        sst_pyr_conn = Connections(_groups['sst_cells'], _groups['pyr_cells'],
                                   equation_builder=istdp_synapse_model,
                                   method=stochastic_decay,
                                   name=group_name+'sst_pyr_conn')
    else:
        pv_pyr_conn = Connections(_groups['pv_cells'], _groups['pyr_cells'],
                                  equation_builder=static_synapse_model(),
                                  method=stochastic_decay,
                                  name=group_name+'pv_pyr_conn')
        sst_pyr_conn = Connections(_groups['sst_cells'], _groups['pyr_cells'],
                                  equation_builder=static_synapse_model(),
                                  method=stochastic_decay,
                                  name=group_name+'sst_pyr_conn')

    # Between interneurons
    pv_pv_conn = Connections(_groups['pv_cells'],
                             _groups['pv_cells'],
                             equation_builder=static_synapse_model(),
                             method=stochastic_decay,
                             name=group_name+'pv_pv_conn')
    # TODO organize alt adp below
    sst_pv_conn = Connections(_groups['sst_cells'],
                              _groups['pv_cells'],
                              equation_builder=altadp_synapse_model(),#static_synapse_model(),
                              method=stochastic_decay,
                              name=group_name+'sst_pv_conn')
    sst_vip_conn = Connections(_groups['sst_cells'],
                               _groups['vip_cells'],
                               equation_builder=static_synapse_model(),
                               method=stochastic_decay,
                               name=group_name+'sst_vip_conn')
    vip_sst_conn = Connections(_groups['vip_cells'],
                               _groups['sst_cells'],
                               equation_builder=static_synapse_model(),
                               method=stochastic_decay,
                               name=group_name+'vip_sst_conn')

    temp_groups = {'pyr_pyr': pyr_pyr_conn,
                   'pyr_pv': pyr_pv_conn,
                   'pyr_sst': pyr_sst_conn,
                   'pyr_vip': pyr_vip_conn,
                   'pv_pyr': pv_pyr_conn,
                   'sst_pyr': sst_pyr_conn,
                   'pv_pv': pv_pv_conn,
                   'sst_pv': sst_pv_conn,
                   'sst_vip': sst_vip_conn,
                   'vip_sst': vip_sst_conn
                   }
    _groups.update(temp_groups)

    # Make connections
    syn_objects = {key: val for key, val in _groups.items() if 'cells' not in key}
    for key, val in syn_objects.items():
        source, target = key.split('_')[0], key.split('_')[1]
        if source==target:
            val.connect('i!=j', p=connectivity_params[key])
        else:
            val.connect(p=connectivity_params[key])

    # Excitatory connections onto somatic compartment
    pyr_pv_conn.set_params(exc_soma_params['static'])
    pyr_sst_conn.set_params(exc_soma_params['static'])
    pyr_vip_conn.set_params(exc_soma_params['static'])

    # Excitatory connections onto dendritic compartment
    pyr_pyr_conn.set_params(exc_dend_params)

    # Inhibitory connections onto somatic compartment
    pv_pv_conn.set_params(inh_soma_params)
    pv_pyr_conn.set_params(inh_soma_params)
    sst_pv_conn.set_params(inh_soma_params)
    sst_vip_conn.set_params(inh_soma_params)
    vip_sst_conn.set_params(inh_soma_params)

    # Inhibitory connections onto dendritic compartment
    sst_pyr_conn.set_params(inh_dend_params)

    # TODO organize alt adp below
    sst_pv_conn.inh_learning_rate = 0.01

    # Delays
    pyr_pyr_conn.delay = np.random.randint(0, 8, size=np.shape(pyr_pyr_conn.j)[0]) * ms
    #feedforward_exc.delay = np.random.randint(0, 8, size=np.shape(feedforward_exc.j)[0]) * ms
    #feedforward_inh.delay = np.random.randint(0, 8, size=np.shape(feedforward_inh.j)[0]) * ms
    #inh_inh_conn.delay = np.random.randint(0, 8, size=np.shape(inh_inh_conn.j)[0]) * ms

    # Random weights initialization
    w_init_group = [pv_pyr_conn, sst_pyr_conn]

    if i_plast == 'plastic_inh' or i_plast == 'plastic_inh0':
        pv_pyr_conn.weight = -1
        sst_pyr_conn.weight = -1
        # 1 = no inhibition, 0 = maximum inhibition
        var_th = 0.50
        add_group_param_init(w_init_group,
                             variable='w_plast',
                             dist_param=synapse_mean_weight['i_e'],
                             scale=1,
                             distribution='normal',
                             clip_min=1,
                             clip_max=15)
        for g in w_init_group:
            g.__setattr__('w_plast', np.array(g.w_plast).astype(int))
    else:
        add_group_param_init(w_init_group,
                             variable='weight',
                             dist_param=synapse_mean_weight['i_e'],
                             scale=1,
                             distribution='normal',
                             unit=-1,
                             clip_min=1,
                             clip_max=15)
        for g in w_init_group:
            g.__setattr__('weight', np.array(g.weight).astype(int))

    # TODO organize alt adp below
    w_init_group = [sst_pv_conn]
    sst_pv_conn.weight = -1
    add_group_param_init(w_init_group,
                         variable='w_plast',
                         dist_param=synapse_mean_weight['i_e'],
                         scale=1,
                         distribution='normal',
                         clip_min=1,
                         clip_max=15)
    for g in w_init_group:
        g.__setattr__('w_plast', np.array(g.w_plast).astype(int))
    w_init_group = [pv_pv_conn, sst_vip_conn, vip_sst_conn]#sst_pv_conn, sst_vip_conn, vip_sst_conn]
    add_group_param_init(w_init_group,
                         variable='weight',
                         dist_param=synapse_mean_weight['i_i'],
                         scale=1,
                         distribution='normal',
                         unit=-1,
                         clip_min=1,
                         clip_max=15)
    for g in w_init_group:
        g.__setattr__('weight', np.array(g.weight).astype(int))

    w_init_group = [pyr_pv_conn, pyr_sst_conn, pyr_vip_conn]
    add_group_param_init(w_init_group,
                         variable='weight',
                         dist_param=synapse_mean_weight['e_i'],
                         scale=1,
                         distribution='normal',
                         clip_min=1,
                         clip_max=15)
    for g in w_init_group:
        g.__setattr__('weight', np.array(g.weight).astype(int))

    w_init_group = [pyr_pyr_conn]
    pyr_pyr_conn.weight = 1
    add_group_param_init(w_init_group,
                         variable='w_plast',
                         dist_param=synapse_mean_weight['e_e'],
                         scale=1,
                         distribution='normal',
                         clip_min=1,
                         clip_max=15)
    for g in w_init_group:
        g.__setattr__('w_plast', np.array(g.w_plast).astype(int))

    generate_mismatch([_groups['pyr_pv'], _groups['pyr_sst'],
                       _groups['pyr_vip'], _groups['pyr_pyr'],
                       _groups['pv_pyr'], _groups['sst_pyr'],
                       _groups['pv_pv'], _groups['sst_pv'],
                       _groups['sst_vip'], _groups['vip_sst']],
                      mismatch_synapse_param)
    generate_mismatch([_groups['pyr_pyr']],
                      mismatch_plastic_param)

    # In case adp is used
    if i_plast == 'plastic_inh':
        # Add proxy activity group
        activity_proxy_group = [_groups['pyr_cells']]
        add_group_activity_proxy(activity_proxy_group,
                                 buffer_size=400,
                                 decay=150)
        pv_pyr_conn.variance_th = np.random.uniform(
            low=var_th - 0.1,
            high=var_th + 0.1,
            size=len(pv_pyr_conn))
        sst_pyr_conn.variance_th = np.random.uniform(
            low=var_th - 0.1,
            high=var_th + 0.1,
            size=len(sst_pyr_conn))
    # TODO organize alt adp below
    from SLIF_run_regs import add_alt_activity_proxy
    add_alt_activity_proxy([_groups['pv_cells']],
                             buffer_size=400,
                             decay=150)

    # Set LFSRs for each group
    #neu_groups = [exc_cells, inh_cells]
    # syn_groups = [exc_exc_conn, exc_inh_conn, inh_exc_conn, feedforward_exc,
    #                 feedforward_inh, inh_inh_conn]
    #ta = create_lfsr(neu_groups, syn_groups, defaultclock.dt)

    return _groups

def generate_mismatch(mismatch_group, mismatch_params, params_unit='ms'):
    """ This functions adds mismatch according to provided dictionary

    Args:
        mismatch_group (list of brian2.groups): Groups that will have mismatch
            added.
        mismatch_params (dict): Mismatch parameters of all elements in the
            group.
        params_unit (str): Indicates the unit of variable. It can be 'ms' or
            'mA'
    """
    if params_unit == 'ms':
        temp_var = 1*ms
    elif params_unit == 'mA':
        temp_var = 1*mA
    else:
        raise('Type unsuported')

    for g in mismatch_group:
        g.add_mismatch(std_dict=mismatch_params)
        # Convert values to integer
        for key in mismatch_params.keys():
            rounded_values = np.around(g.__getattr__(key)/temp_var)
            if np.any(rounded_values):
                rounded_values = np.clip(rounded_values, 0, np.max(rounded_values))
            g.__setattr__(key, rounded_values*temp_var)
