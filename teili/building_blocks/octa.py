#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:58:50 2019

@author: matteo, mmilde
"""

from brian2 import ms, pA
from brian2 import SpikeGeneratorGroup, SpikeMonitor
from brian2 import prefs, implementation, check_units
import numpy as np

from teili import TeiliNetwork
from teili.core.groups import Neurons, Connections
from teili.core import tags as tags

from teili.models.synapse_models import DPISyn, DPIstdp, DPIadp, DPIstdgm
from teili.models.neuron_models import OCTA_Neuron as octa_neuron
from teili.building_blocks.building_block import BuildingBlock
from teili.building_blocks.wta import WTA
from teili.stimuli.testbench import WTA_Testbench, OCTA_Testbench
from teili.models.parameters.octa_params import wta_params, octa_params,\
    mismatch_neuron_param, mismatch_synap_param
from teili.tools.group_tools import add_goup_weight_init,\
    add_group_weight_decay, add_group_weight_re_init,\
    add_group_activity_proxy
from teili.tools.bb_tools import add_bb_mismatch
from teili.tools.io import save_monitor, load_monitor,\
    save_weights, load_weights

prefs.codegen.target = "numpy"
@implementation('numpy', discard_units=True)
@check_units(Ipred_plast=1, source_N=1, target_N=1, re_init_threshold=1, result=1)
def re_init_ipred(Ipred_plast, source_N, target_N, re_init_threshold=0.2):
    data = np.zeros((source_N, target_N))
    data = np.reshape(Ipred_plast, (source_N, target_N))

    reinit_index = np.mean(data, 1) > (1 - re_init_threshold)

    data[reinit_index, :] = np.zeros((np.sum(reinit_index), target_N))
    data = np.clip(data, 0, 1)
    return data.flatten()


class Octa(BuildingBlock):
    """The Online Clustering of Temporal Activity (OCTA) `BuildingBlock`.
    This neural algorithm is designed to cluster incoming spatio-temporal
    patterns of asynchronous streams of events as provided by event-driven
    sensors such as the Dynamic Vision Sensor (DVS).

    This class provides a basic hierarchical `BuildingBlock` called OCTA.
    For more information please refer to our [Documentation](https://teili.readthedocs.io/en/latest/scripts/Building%20Blocks.html#online-clustering-of-temporal-activity-octa)
    This is a level 2 hierarchical building block, it uses basic building blocks such
    as the WTA.

    If you want to change the default parameters of your building block
        you need to define a dictionary, which you pass to the building_block:
    """

    def __init__(self, name='octa*',
                 neuron_eq_builder=octa_neuron,
                 wta_params=wta_params,
                 octa_params=octa_params,
                 num_input_neurons=10,
                 num_neurons=7,
                 external_input=True,
                 noise=True,
                 monitor=True,
                 debug=False):
        """Initializes building block object with defined
        connectivity scheme.

        Args:
            name (str): Name of the OCTA BuildingBlock.
            neuron_eq_builder (class, optional): neuron class as imported from
                models/neuron_models.
            wta_params (Dict, optional): WTA parameter dictionary.
            octa_params (Dict, optional): OCTA parameter dictionary.
            num_input_neurons (int, optional): Size of input population. The
                default is set to 10 as OCTA was originally designed to learn
                visual receptive fields from 10 x 10 patches. Note that the
                number provided will be internally squared, meaning that
                `num_input_neurons=10` will result in an input of size 10*10.
            num_neurons (int, optional): Size of WTA neuron population.
                Theoretically this number can be bigger than
                `num_input_neurons`. However, as the objective is to cluster,
                the number is set to be by default 7, i.e. smaller than
                `num_input_neurons`.
            external_input (bool, optional): Flag to include an input in the
                form of a rotating bar.
            noise (bool, optional): Flag to include a noise source.
            monitor (bool, optional): Flag to auto-generate spike and state
                monitors.
            debug (bool, optional): Flag to gain additional information.
        """

        block_params = {}
        block_params.update(octa_params)

        BuildingBlock.__init__(self,
                               name,
                               neuron_eq_builder,
                               block_params,
                               debug,
                               monitor)

        self.sub_blocks,\
            self._groups,\
            self.monitors,\
            self.standalone_params = gen_octa(name,
                                              neuron_eq_builder,
                                              num_neurons=num_neurons,
                                              num_input_neurons=num_input_neurons,
                                              external_input=external_input,
                                              wta_params=wta_params,
                                              noise=noise,
                                              monitor=monitor,
                                              debug=debug,
                                              **block_params)

        # Defining input - output - hidden groups
        self.input_groups.update({'projection': self._groups['n_proj']})

        self.output_groups.update({
            'compression': self.sub_blocks['compression']._groups['n_exc'],
            'prediction': self.sub_blocks['prediction']._groups['n_exc']})

        self.hidden_groups.update({
            'comp_n_inh': self.sub_blocks['compression']._groups['n_inh'],
            'pred_n_inh': self.sub_blocks['prediction']._groups['n_inh']})

        set_OCTA_tags(self, self._groups)


def gen_octa(groupname,
             neuron_eq_builder=octa_neuron,
             num_input_neurons=10,
             num_neurons=7,
             wta_params=wta_params,
             distribution='gamma',
             dist_param_init=0.5,
             scale_init=1.0,
             dist_param_re_init=0.4,
             scale_re_init=0.9,
             re_init_index=None,
             re_init_threshold=0.2,
             buffer_size_plast=200,
             noise_weight=30.0,
             variance_th_c=0.5,
             variance_th_p=0.4,
             learning_rate=0.007,
             inh_learning_rate=0.01,
             decay=150,
             tau_stdp=10 * ms,
             tau_pred=1.5 * ms,
             seed=42,
             external_input=True,
             noise=True,
             monitor=True,
             debug=False):
    """ Generates an OCTA `BuildingBlock`.

    Args:
            groupname (str): Name of the OCTA BuildingBlock.
            neuron_eq_builder (class, optional): Neuron model class as
                imported from teili.models.neuron_models.
            num_input_neurons (int, optional): Number of neurons of the input.
                Note that this number will be squared, e.g. 10 --> 100 input
                neurons.
            num_neurons (int, optional): Number of neurons in the compression
                layer. Note that this number will be squared, e.g. 7 --> 49
                neurons.
            wta_params (dict, optional): Dictionary of parameters for the
                WTA `BuildingBlock`.
            distribution (str, optional): Parameter to determine the random
                distribution to be used to initialise the weights. Possible
                values are 'gaussian' or 'gamma'.
            dist_param_init (float, optional): Mean of distribution to
                initialise the synaptic weights in case of 'gaussian' or
                shape parameter k for 'gamma' distribution.
            scale_init (float, optional): Standard deviation of distribution
                to initialise synaptic weights in case of 'gaussian' or
                scale parameter sigma for 'gamma' distribution.
            dist_param_re_init (float, optional): Mean of distribution to
                re-initialise synaptic weights in case of 'gaussian' or
                shape parameter k for 'gamma' distribution.
            scale_re_init (float, optional): Standard deviation of
                distribution to re-initialise synaptic weights in case of
                'gaussian' or scale parameter sigma for 'gamma' distribution.
            re_init_threshold (float, optional): Threshold below which
                weights are re-initialised.
            buffer_size_plast (int, optional): Parameter to set the size of
                the buffer for activity dependent plasticity rule. The size
                of the buffer determines how many samples are considered in
                order to calculate the activity proxy of post synaptic
                neurons.
            noise_weight (float, optional): Synaptic strength of Poisson
                noise SpikeGenerator.
            variance_th_c (float, optional): Mean of distribution to
                initialise the reference values for activity dependent
                plasticity of the compression layer. The reference values are
                sampled from a uniform distribution.
            variance_th_p (float, optional): Mean of distribution to
                initialise the reference values for activity dependent
                plasticity of the prediction layer. The reference values are
                sampled from a uniform distribution.
            learning_rate (float, optional): Learning rate for excitatory
                synapses.
            inh_learning_rate (float, optional): Learning rate for inhibitory
                synapses.
            decay (int, optional): Time constant for decay of exponentionally
                weighted activity proxy.
            decay_strategy (str, optional): Weight decay strategy. Either
                'global' which decays weight based on a fixed time interval,
                on 'local' which performs event-driven weight decay.
            tau_stdp (float ms, optional): Time constant for STDP window.
            seed (int, optional): Seed to initialise random distribution
                from which the mismatch is sampled.
            external_input (bool, optional): Flag to connect an external
                input to the OCTA block. If set to `False` the user is
                expected to connect another neuron population.
            noise (bool, optional): Flag to connect poisson noise generator.
            monitor (bool, optional): Flag to spawn SpikeMonitors.
            debug (bool, optional): Flag to print more debug information.

    Returns:
        sub_blocks (dictionary): Keys to all sub_blocks of the network
        _groups (dictionary): Keys to all neuron and synapse groups
            specific to this BB.
        monitors (dictionary): Keys to all spike and state monitors specific
            to this BB.
        standalone_params (dictionary): Not implemented at the moment as
            OCTA depends on `run_regularly` functions which are not
            supported.

    Functional information about the network:
        The OCTA network is an implementation of the canonical microcircuit
        found in the cortex leveraging temporal information to extract
        meaning from the input data. It consists of two WTA networks
        (compression and prediction) connected in a recurrent manner
        (figure of connectivity can be found in the docs).
        Every building block in the teili implementation has a cortical
        counterpart for which the connectivity and function is preserved:
            compression['n_proj'] : Layer 4
            compression['n_exc'] : Layer 2/3
            prediction['n_exc'] : Layer 5/6

        Given a high dimensional input in L2/3, the network extracts in the
        recurrent connections of L4 a lower dimensional representation of
        temporal dependencies by learning spatio-temporal features.
    """
    if debug:
        print("Creating OCTA BuildingBlock!")
        print("External input: ", external_input)
        print("Noise: ", noise)
        print("Monitor: ", monitor)

    num_inh_neurons_c = int(num_neurons**2 / 4)
    num_inh_neurons_p = int(num_input_neurons**2 / 4)

    compression = WTA(name='compression',
                      dimensions=2,
                      neuron_eq_builder=neuron_eq_builder,
                      num_neurons=num_neurons,
                      num_inh_neurons=num_inh_neurons_c,
                      num_input_neurons=num_input_neurons,
                      num_inputs=4,
                      block_params=wta_params,
                      monitor=monitor)

    prediction = WTA(name='prediction',
                     dimensions=2,
                     neuron_eq_builder=neuron_eq_builder,
                     num_neurons=num_input_neurons,
                     num_inh_neurons=num_inh_neurons_p,
                     num_input_neurons=num_neurons,
                     num_inputs=4,
                     block_params=wta_params,
                     monitor=monitor)

    # Define a projection layer which recieves external input and relays it
    # to the compression WTA with the s_inp_exc
    projection = Neurons(N=num_input_neurons**2,
                         equation_builder=neuron_eq_builder(num_inputs=4),
                         refractory=wta_params['rp_exc'],
                         name=groupname + '_projection')

    compression._groups['s_inp_exc'] = Connections(projection,
                                                   compression._groups['n_exc'],
                                                   equation_builder=DPIstdp,
                                                   method='euler',
                                                   name='s_proj_comp')

    compression._groups['s_inp_exc'].connect(True)
    compression._set_tags(tags.basic_wta_s_inp_exc,
                          compression._groups['s_inp_exc'])
    compression._groups['s_inp_exc']._tags['level'] = 2
    compression._groups['s_inp_exc']._tags['sign'] = 'exc'
    compression._groups['s_inp_exc']._tags['bb_type'] = 'OCTA'
    compression._groups['s_inp_exc'].weight = wta_params['we_inp_exc']
    compression._groups['s_inp_exc'].taupre = tau_stdp
    compression._groups['s_inp_exc'].taupost = tau_stdp

    # Change the equation model of the recurrent connections in
    # compression WTA
    replace_connection(compression, 'n_exc',
                       compression, 'n_exc',
                       's_exc_exc',
                       equation_builder=DPIstdp(),
                       name='compression' + '_s_exc_exc')

    compression._set_tags(tags.basic_wta_s_exc_exc,
                          compression._groups['s_exc_exc'])
    compression._groups['s_exc_exc'].weight = wta_params['we_exc_exc']

    # Change the eqaution model to include adaptation on the reccurrent
    # connection of the compression WTA.
    replace_connection(compression, 'n_inh',
                       compression, 'n_exc',
                       's_inh_exc',
                       equation_builder=DPIadp())

    compression._set_tags(tags.basic_wta_s_inh_exc,
                          compression._groups['s_inh_exc'])

    compression._groups['s_inh_exc'].weight = wta_params['wi_inh_exc']
    compression._groups['s_inh_exc'].variance_th = np.random.uniform(
        low=variance_th_c - 0.1,
        high=variance_th_c + 0.1,
        size=len(compression._groups['s_inh_exc']))

    # Change the eqaution model to include adaptation on the reccurrent
    # connection of the prediction WTA.
    replace_connection(prediction, 'n_inh',
                       prediction, 'n_exc',
                       's_inh_exc',
                       equation_builder=DPIadp())

    projection.tau_pred = tau_pred

    prediction._set_tags(tags.basic_wta_s_inh_exc,
                         prediction._groups['s_inh_exc'])
    prediction._groups['s_inh_exc'].weight = wta_params['wi_inh_exc']
    prediction._groups['s_inh_exc'].variance_th = np.random.uniform(
        low=variance_th_p - 0.1,
        high=variance_th_p + 0.1,
        size=len(prediction._groups['s_inh_exc']))

    # Connect the compression WTA and the prediction WTA
    replace_connection(compression, 'n_exc',
                       prediction, 'n_exc',
                       's_inp_exc',
                       equation_builder=DPIstdp())

    prediction._set_tags(tags.basic_wta_s_inp_exc,
                         prediction._groups['s_inp_exc'])
    prediction._groups['s_inp_exc']._tags['sign'] = 'exc'
    prediction._groups['s_inp_exc']._tags['bb_type'] = 'OCTA'
    prediction._groups['s_inp_exc']._tags['level'] = 2
    prediction._groups['s_inp_exc'].weight = wta_params['we_inp_exc']
    prediction._groups['s_inp_exc'].taupre = tau_stdp
    prediction._groups['s_inp_exc'].taupost = tau_stdp

    # Include stdp in recurrent connections in prediction WTA
    replace_connection(prediction, 'n_exc',
                       prediction, 'n_exc',
                       's_exc_exc',
                       equation_builder=DPIstdp())

    compression._set_tags(tags.basic_wta_s_exc_exc,
                          prediction._groups['s_exc_exc'])
    prediction._groups['s_exc_exc'].weight = wta_params['we_exc_exc']

    # Connect the projection layer to the prediction WTA - can be seen as an
    # error connection
    s_proj_pred = Connections(projection,
                              prediction._groups['n_exc'],
                              equation_builder=DPIstdp(),
                              method='euler',
                              name=groupname + '_s_proj_pred')

    s_proj_pred.connect('True')
    s_proj_pred.weight = wta_params['we_inp_exc']
    s_proj_pred.taupre = tau_stdp
    s_proj_pred.taupost = tau_stdp

    # Connect the prediction WTA to the projection layer - can be seen as a
    # predictive connection
    s_pred_proj = Connections(prediction._groups['n_exc'],
                              projection,
                              equation_builder=DPIstdgm(),
                              method='euler',
                              name=groupname + '_s_pred_proj')

    s_pred_proj.connect(True)
    s_pred_proj.Ipred_plast = np.zeros((len(s_pred_proj)))
    s_pred_proj.tau_pred = tau_pred

    # Set learning rate
    compression._groups['s_inp_exc'].dApre = learning_rate
    compression._groups['s_exc_exc'].dApre = learning_rate
    prediction._groups['s_inp_exc'].dApre = learning_rate
    prediction._groups['s_exc_exc'].dApre = learning_rate

    s_proj_pred.dApre = learning_rate
    s_pred_proj.dApre = learning_rate

    compression._groups['s_inh_exc'].inh_learning_rate = inh_learning_rate
    prediction._groups['s_inh_exc'].inh_learning_rate = inh_learning_rate

    # List of groups which need the same run regularly function to be added
    w_init_group = [compression._groups['s_exc_exc']] +\
                   [compression._groups['s_inp_exc']] +\
                   [compression._groups['s_inh_exc']] +\
                   [prediction._groups['s_inh_exc']] +\
                   [prediction._groups['s_inp_exc']] +\
                   [prediction._groups['s_exc_exc']] +\
                   [s_proj_pred]

    add_goup_weight_init(w_init_group,
                         dist_param=dist_param_init,
                         scale=scale_init,
                         distribution=distribution)

    weight_decay_group = [compression._groups['s_inp_exc']] +\
                         [compression._groups['s_exc_exc']] +\
                         [prediction._groups['s_inp_exc']] +\
                         [prediction._groups['s_exc_exc']] +\
                         [s_proj_pred]

    add_group_weight_decay(weight_decay_group,
                           decay_rate=0.999,
                           dt=100*ms)

    s_pred_proj.add_state_variable('decay')
    s_pred_proj.decay = 0.999
    dict_append = {'weight decay': 'clock-driven'}
    s_pred_proj._tags.update(dict_append)

    s_pred_proj.run_regularly('''Ipred_plast *= decay''', dt=100*ms)

    weight_re_init_group = [compression._groups['s_inp_exc']] +\
                           [compression._groups['s_exc_exc']] +\
                           [prediction._groups['s_inp_exc']] +\
                           [prediction._groups['s_exc_exc']] +\
                           [s_proj_pred]

    add_group_weight_re_init(weight_re_init_group,
                             re_init_index=re_init_index,
                             re_init_threshold=re_init_threshold,
                             dist_param_re_init=dist_param_re_init,
                             scale_re_init=scale_re_init,
                             distribution=distribution)

    s_pred_proj.namespace.update({'re_init_ipred': re_init_ipred})
    s_pred_proj.namespace['re_init_threshold'] = re_init_threshold

    s_pred_proj.run_regularly('''Ipred_plast = re_init_ipred(Ipred_plast,\
                                                             N_pre,\
                                                             N_post,\
                                                             re_init_threshold)''',
                              dt=50 * ms)

    activity_proxy_group = [compression._groups['n_exc']] + \
                           [prediction._groups['n_exc']]

    add_group_activity_proxy(activity_proxy_group,
                             buffer_size=buffer_size_plast,
                             decay=decay)

    # initialize neuron parameters
    compression._groups['n_exc'].Imem = 0.5 * pA
    compression._groups['n_inh'].Imem = 0.5 * pA
    prediction._groups['n_exc'].Imem = 0.5 * pA
    prediction._groups['n_inh'].Imem = 0.5 * pA

    # initialize mismatch
    add_bb_mismatch(compression, seed)
    add_bb_mismatch(prediction, seed)
    s_proj_pred.add_mismatch(mismatch_synap_param, seed=seed)
    projection.add_mismatch(mismatch_neuron_param, seed=seed)

    if external_input is True:
        spike_gen = SpikeGeneratorGroup(N=num_input_neurons**2,
                                        indices=[],
                                        times=[] * ms)

        s_inp_exc = Connections(spike_gen, projection,
                                equation_builder=DPISyn(),
                                method='euler',
                                name=groupname + '_s_inp_exc')

        s_inp_exc.connect('i==j')
        s_inp_exc.weight = 3250.

    if noise:
        testbench_c = WTA_Testbench()
        testbench_p = WTA_Testbench()

        testbench_c.background_noise(num_neurons=num_neurons, rate=10)
        testbench_p.background_noise(num_neurons=num_input_neurons, rate=10)

        noise_syn_c_exc = Connections(testbench_c.noise_input,
                                      compression._groups['n_exc'],
                                      equation_builder=DPISyn(),
                                      name=groupname + '_noise_comp_exc')

        noise_syn_c_exc.connect("i==j")
        noise_syn_c_exc.weight = noise_weight

        noise_syn_p_exc = Connections(testbench_c.noise_input,
                                      prediction._groups['n_exc'],
                                      equation_builder=DPISyn(),
                                      name=groupname + '_noise_pred_exc')

        noise_syn_p_exc.connect("i==j")

        noise_syn_p_exc.weight = noise_weight
        compression._groups['n_exc']._tags['noise'] = 1
        prediction._groups['n_exc']._tags['noise'] = 1

        compression._groups['n_exc']._tags['num_inputs'] = compression._groups[
            'n_exc']._tags['num_inputs'] + 1
        prediction._groups['n_exc']._tags['num_inputs'] = prediction._groups[
            'n_exc']._tags['num_inputs'] + 1

    _groups = {'s_proj_pred': s_proj_pred,
               's_pred_proj': s_pred_proj,
               's_comp_pred': prediction._groups['s_inp_exc'],
               's_proj_comp': compression._groups['s_inp_exc'],
               'n_proj': projection}

    if external_input is True:
        ext_int_dict = {'s_inp_exc': s_inp_exc,
                        'spike_gen': spike_gen}
        _groups.update(ext_int_dict)

    if noise:
        noise_syn = {'pred_noise_syn_exc': noise_syn_p_exc,
                     'comp_noise_syn_exc': noise_syn_c_exc,
                     'pred_noise_gen': testbench_p.noise_input,
                     'comp_noise_gen': testbench_c.noise_input
                     }
        _groups.update(noise_syn)

    monitors = {}
    if monitor:
        spikemon_proj = SpikeMonitor(projection,
                                     name='spikemon_proj')

        monitors = {'spikemon_proj': spikemon_proj}

        if external_input:
            spikemon_inp = SpikeMonitor(spike_gen,
                                        name=groupname + '_spike_gen')

            monitors.update({'spikemon_inp': spikemon_inp})

    if debug:
        print('The keys of the ' + groupname + ' output dict are:')
        for key in _groups:
            print(key)

    standalone_params = {}

    sub_blocks = {'compression': compression,
                  'prediction': prediction}

    return sub_blocks, _groups, monitors, standalone_params


def replace_connection(bb_source, population_source,
                       bb_target, population_target,
                       connection_name,
                       equation_builder,
                       method='euler', name=None):
    '''
    This function replaces and adds a new connection between two groups

    Args:
        bb_source (Obj): The source BuildingBlock object from wich the
            connection should be updated
        population_source (str): The source population
        bb_target (Obj): The target BuildingBlock object from wich the
            connection should be updated
        population_target (str): The target population
        connection_name (str): The name of the connection
        equation_builder (obj): The equation_builder object which holds the
            synapse model
        method (str, optional): Integration method, default is 'euler'
        name (str, optional): If name of synapse differs from
            connection_name.
    '''

    if name is None:
        name = bb_target._groups[connection_name].name

    bb_target._groups[connection_name] = Connections(bb_source._groups[
        population_source],
        bb_target._groups[
        population_target],
        equation_builder=equation_builder,
        method=method,
        name=name)
    bb_target._groups[connection_name].connect(True)

    return None


def set_OCTA_tags(self, _groups):
    '''Sets default tags to a OCTA network

    Args:
        _groups (Dictionary): Dictionary of all groups belonging to the
            BuildingBlock
    '''
    self._set_tags(tags.basic_octa_s_proj_pred, _groups['s_proj_pred'])
    self._set_tags(tags.basic_octa_s_pred_proj, _groups['s_pred_proj'])
    self._set_tags(tags.basic_octa_n_sg, _groups['spike_gen'])
    self._set_tags(tags.basic_octa_pred_noise_sg, _groups['pred_noise_gen'])
    self._set_tags(tags.basic_octa_comp_noise_sg, _groups['comp_noise_gen'])
    self._set_tags(tags.basic_octa_n_proj, _groups['n_proj'])
    self._set_tags(tags.basic_octa_s_pred_noise,
                   _groups['pred_noise_syn_exc'])
    self._set_tags(tags.basic_octa_s_comp_noise,
                   _groups['comp_noise_syn_exc'])
