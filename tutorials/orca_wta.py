""" This module can be used to scale-up models published by Wang et al. (2018).
    """

import os
import numpy as np

from brian2 import ms, Hz, defaultclock, ExplicitStateUpdater, PoissonGroup

from teili.building_blocks.building_block import BuildingBlock
from teili.core.groups import Neurons, Connections
from teili.models.neuron_models import QuantStochLIF as static_neuron_model
from teili.models.synapse_models import QuantStochSyn as static_synapse_model
from teili.models.synapse_models import QuantStochSynStdp as stdp_synapse_model
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder

from orca_params import connection_probability_old, excitatory_neurons,\
    inhibitory_neurons, excitatory_plastic_syn, inhibitory_syn
from teili.tools.misc import neuron_group_from_spikes

# Load other models
path = os.path.expanduser("/home/pablo/git/teili")
model_path = os.path.join(path, "teili", "models", "equations", "")
adp_synapse_model = SynapseEquationBuilder.import_eq(
        model_path + 'StochSynAdp.py')
adp_synapse_model0 = SynapseEquationBuilder.import_eq(
        model_path + 'StochSynAdp0.py')
adapt_neuron_model = NeuronEquationBuilder(base_unit='quantized',
        intrinsic_excitability='threshold_adaptation',
        position='spatial')
reinit_synapse_model = SynapseEquationBuilder(base_unit='quantized',
        plasticity='quantized_stochastic_stdp',
        structural_plasticity='stochastic_counter')

defaultclock.dt = 1 * ms
stochastic_decay = ExplicitStateUpdater('''x_new = f(x,t)''')

class ORCA_WTA(BuildingBlock):
    """A WTA with diverse inhibitory population. This could represent a single
       layer in a cortical sheet.

    Attributes:
        

    """

    def __init__(self,
                 input_indices,
                 input_times,
                 name='orca_wta*',
                 connectivity_params=connection_probability_old,
                 exc_neu_params=excitatory_neurons,
                 inh_neu_params=inhibitory_neurons,
                 exc_plastic_params=excitatory_plastic_syn,
                 inh_params=inhibitory_syn,
                 verbose=False,
                 monitor=False,
                 num_exc_neurons=200,
                 ratio_pv=.46,
                 ratio_sst=.36,
                 ratio_vip=.18,
                 noise=False):
        """Initializes ORCA WTA building block

        Args:
            input_indices (numpy.array): Indices of the original source.
            input_times (numpy.array): Time stamps with unit of original spikes
                in ms.
            name (str, required): Name of the building_block population
            connectivity_params (dict): Dictionary which holds connectivity
                parameters
            exc_neu_params (dict): Dictionary which holds parameters of
                excitatory neurons
            inh_neu_params (dict): Dictionary which holds parameters of
                inhibitory neurons 
            exc_plastic_params (dict): Dictionary which holds parameters
                of excitatory plastic connections
            inh_params (dict): Dictionary which holds parameters
                of inhibitory static connections
            verbose (bool, optional): Flag to gain additional information
            monitor (bool, optional): Flag to auto-generate spike and state
                monitors
            num_exc_neurons (int, optional): Size of excitatory population.
            ratio_pv (float, optional): Fraction of inhibitory neurons that
                are PV cells.
            ratio_sst (float, optional): Fraction of inhibitory neurons that
                are SST cells.
            ratio_vip (float, optional): Fraction of inhibitory neurons that
                are VIP cells.
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
        self._groups = gen_orca(num_exc_neurons=num_exc_neurons,
                                input_indices=input_indices,
                                input_times=input_times,
                                ratio_pv=ratio_pv,
                                ratio_sst=ratio_sst,
                                ratio_vip=ratio_vip,
                                monitor=monitor,
                                verbose=verbose,
                                connectivity_params=connectivity_params,
                                exc_neu_params=exc_neu_params,
                                inh_neu_params=inh_neu_params,
                                exc_plastic_params=exc_plastic_params,
                                inh_params=inh_params,
                                noise=noise)

def gen_orca(num_exc_neurons,
             input_indices,
             input_times,
             ratio_pv,
             ratio_sst,
             ratio_vip,
             connectivity_params,
             exc_neu_params,
             inh_neu_params,
             exc_plastic_params,
             inh_params,
             monitor,
             verbose,
             noise):
    """Generates network with specified characteristics and elements described
    by Wang et al. (2018).

    Args:
        input_indices (numpy.array): Indices of the original source.
        input_times (numpy.array): Time stamps with unit of original spikes
            in ms.
        name (str, required): Name of the building_block population
        connectivity_params (dict): Dictionary which holds building_block
            specific parameters
        exc_neu_params (dict): Dictionary which holds parameters of
            excitatory neurons
        inh_neu_params (dict): Dictionary which holds parameters of
            inhibitory neurons 
        exc_plastic_params (dict): Dictionary which holds parameters
            of excitatory plastic connections
        inh_params (dict): Dictionary which holds parameters
            of inhibitory static connections
        verbose (bool, optional): Flag to gain additional information
        monitor (bool, optional): Flag to auto-generate spike and state
            monitors
        num_exc_neurons (int, optional): Size of excitatory population.
        ratio_pv (float, optional): Fraction of inhibitory neurons that
            are PV cells.
        ratio_sst (float, optional): Fraction of inhibitory neurons that
            are SST cells.
        ratio_vip (float, optional): Fraction of inhibitory neurons that
            are VIP cells.
        noise (bool, optional): Flag to determine if background noise is to
            be added to neurons. This is generated with a poisson process.
    """
    # TODO remove when no longer testing, as well as if'
    i_plast = 'plastic_inh0'
    # Convert input into neuron group (necessary for STDP compatibility)
    sim_duration = input_times[-1]
    num_input = max(input_indices)
    seq_cells = neuron_group_from_spikes(num_input,
                                         defaultclock.dt,
                                         sim_duration,
                                         spike_indices=input_indices,
                                         spike_times=input_times)

    # Creating populations
    num_inh = int(num_exc_neurons/4)
    #num_inh = int(num_exc_neurons/1.6) 
    num_pv = int(num_inh * ratio_pv)
    num_sst = int(num_inh * ratio_sst)
    num_vip = int(num_inh * ratio_vip)

    pyr_cells = Neurons(num_exc_neurons,
                        equation_builder=adapt_neuron_model(num_inputs=4),
                        method=stochastic_decay,
                        name='pyr_cells',
                        verbose=True)

    if i_plast == 'plastic_inh':
        dummy_unit = 1*mV
        pyr_cells.variables.add_array('activity_proxy',
                                       size=pyr_cells.N,
                                       dimensions=dummy_unit.dim)
        pyr_cells.variables.add_array('normalized_activity_proxy',
                                       size=pyr_cells.N)

    pv_cells = Neurons(num_pv,
                       equation_builder=static_neuron_model(num_inputs=4),
                       method=stochastic_decay,
                       name='pv_cells',
                       verbose=True)
    sst_cells = Neurons(num_sst,
                        equation_builder=static_neuron_model(num_inputs=3),
                        method=stochastic_decay,
                        name='sst_cells',
                        verbose=True)
    vip_cells = Neurons(num_vip,
                        equation_builder=static_neuron_model(num_inputs=3),
                        method=stochastic_decay,
                        name='vip_cells',
                        verbose=True)

    # Creating connections
    # Connecting inputs
    input_pyr_conn = Connections(seq_cells, pyr_cells,
            equation_builder=reinit_synapse_model(),
            method=stochastic_decay,
            name='input_pyr_conn')
    input_pv_conn = Connections(seq_cells, pv_cells,
            equation_builder=static_synapse_model(),
            method=stochastic_decay,
            name='input_pv_conn')
    input_sst_conn = Connections(seq_cells, sst_cells,
            equation_builder=static_synapse_model(),
            method=stochastic_decay,
            name='input_sst_conn')
    input_vip_conn = Connections(seq_cells, vip_cells,
            equation_builder=static_synapse_model(),
            method=stochastic_decay,
            name='input_vip_conn')

    # Pyramidal to interneurons
    pyr_pyr_conn = Connections(pyr_cells, pyr_cells,
            equation_builder=stdp_synapse_model(),
            method=stochastic_decay,
            name='pyr_pyr_conn')
    pyr_pv_conn = Connections(pyr_cells, pv_cells,
            equation_builder=static_synapse_model(),
            method=stochastic_decay,
            name='pyr_pv_conn')
    pyr_sst_conn = Connections(pyr_cells, sst_cells,
            equation_builder=static_synapse_model(),
            method=stochastic_decay,
            name='pyr_sst_conn')
    pyr_vip_conn = Connections(pyr_cells, vip_cells,
            equation_builder=static_synapse_model(),
            method=stochastic_decay,
            name='pyr_vip_conn')

    # Interneurons to pyramidal
    if i_plast == 'plastic_inh':
        pv_pyr_conn = Connections(pv_cells, pyr_cells,
                equation_builder=adp_synapse_model,
                method=stochastic_decay,
                name='pv_pyr_conn')
        sst_pyr_conn = Connections(sst_cells, pyr_cells,
                                   equation_builder=adp_synapse_model,
                                   method=stochastic_decay,
                                   name='sst_pyr_conn')
    elif i_plast == 'plastic_inh0':
        pv_pyr_conn = Connections(pv_cells, pyr_cells,
                                  equation_builder=adp_synapse_model0,
                                  method=stochastic_decay,
                                  name='pv_pyr_conn')
        sst_pyr_conn = Connections(sst_cells, pyr_cells,
                                   equation_builder=adp_synapse_model0,
                                   method=stochastic_decay,
                                   name='sst_pyr_conn')
    else:
        pv_pyr_conn = Connections(pv_cells, pyr_cells,
                                  equation_builder=static_synapse_model(),
                                  method=stochastic_decay,
                                  name='pv_pyr_conn')
        sst_pyr_conn = Connections(sst_cells, pyr_cells,
                                  equation_builder=static_synapse_model(),
                                  method=stochastic_decay,
                                  name='sst_pyr_conn')

    # Between interneurons
    pv_pv_conn = Connections(pv_cells, pv_cells,
            equation_builder=static_synapse_model(),
            method=stochastic_decay,
            name='pv_pv_conn')
    sst_pv_conn = Connections(sst_cells, pv_cells,
            equation_builder=static_synapse_model(),
            method=stochastic_decay,
            name='sst_pv_conn')
    sst_vip_conn = Connections(sst_cells, vip_cells,
            equation_builder=static_synapse_model(),
            method=stochastic_decay,
            name='sst_vip_conn')
    vip_sst_conn = Connections(vip_cells, sst_cells,
            equation_builder=static_synapse_model(),
            method=stochastic_decay,
            name='vip_sst_conn')

    if noise:
        rate_distribution = np.random.randint(1, 15, size=num_exc_neurons) * Hz
        poisson_activity = PoissonGroup(num_exc_neurons, rate_distribution)
        background_activity = Connections(poisson_activity, pyr_cells,
                                          equation_builder=static_synapse_model(),
                                          method=stochastic_decay,
                                          name='background_activity')
    _groups = {
            'pyr_cells': pyr_cells,
            'pv_cells': pv_cells,
            'sst_cells': sst_cells,
            'vip_cells': vip_cells,
            'input_pyr': input_pyr_conn,
            'input_pv': input_pv_conn,
            'input_sst': input_sst_conn,
            'input_vip': input_vip_conn,
            'pyr_pyr': pyr_pyr_conn,
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

    # Make connections and set sparsity (required for structural plasticity)
    syn_objects = {key: val for key, val in _groups.items() if 'cells' not in key}
    for key, val in syn_objects.items():
        if key == 'pyr_pyr':
            val.connect('i!=j', p=connectivity_params[key])
        else:
            val.connect(p=connectivity_params[key])

    if noise:
        background_activity.connect('i==j')
        background_activity.tausyn = 3*ms

    # Introduce sparsity on input channels
    for neu in range(num_exc_neurons):
        ffe_zero_w = np.random.choice(num_input, int(num_input*.3), replace=False)
        input_pyr_conn.weight[ffe_zero_w, neu] = 0
        input_pyr_conn.w_plast[ffe_zero_w, neu] = 0

    # Set general parameters and mismatch
    pyr_cells.set_params(exc_neu_params)
    pv_cells.set_params(inh_neu_params)
    sst_cells.set_params(inh_neu_params)
    vip_cells.set_params(inh_neu_params)

    input_pyr_conn.set_params(exc_plastic_params)
    input_pv_conn.set_params(exc_plastic_params)
    input_sst_conn.set_params(exc_plastic_params)
    input_vip_conn.set_params(exc_plastic_params)
    pyr_pyr_conn.set_params(exc_plastic_params)
    pyr_pv_conn.set_params(exc_plastic_params)
    pyr_sst_conn.set_params(exc_plastic_params)
    pyr_vip_conn.set_params(exc_plastic_params)
    pv_pv_conn.set_params(inh_params)
    pv_pyr_conn.set_params(inh_params)
    sst_pyr_conn.set_params(inh_params)
    sst_pv_conn.set_params(inh_params)
    sst_vip_conn.set_params(inh_params)
    vip_sst_conn.set_params(inh_params)

    if i_plast == 'plastic_inh' or i_plast == 'plastic_inh0':
        pv_pyr_conn.inh_learning_rate = 0.01
        sst_pyr_conn.inh_learning_rate = 0.01

    # Delays
    pyr_pyr_conn.delay = np.random.randint(0, 8, size=np.shape(pyr_pyr_conn.j)[0]) * ms
    #feedforward_exc.delay = np.random.randint(0, 8, size=np.shape(feedforward_exc.j)[0]) * ms
    #feedforward_inh.delay = np.random.randint(0, 8, size=np.shape(feedforward_inh.j)[0]) * ms
    #inh_inh_conn.delay = np.random.randint(0, 8, size=np.shape(inh_inh_conn.j)[0]) * ms

    return _groups
