""" This module can be used to scale-up models published by Wang et al. (2018).
    """

import os
import numpy as np

from brian2 import ms, Hz, mV, ohm, defaultclock, ExplicitStateUpdater,\
        PoissonGroup

from teili.building_blocks.building_block import BuildingBlock
from teili.core.groups import Neurons, Connections
from teili.models.neuron_models import QuantStochLIF as static_neuron_model
from teili.models.synapse_models import QuantStochSyn as static_synapse_model
from teili.models.synapse_models import QuantStochSynStdp as stdp_synapse_model
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.tools.group_tools import add_group_activity_proxy,\
    add_group_params_re_init, add_group_param_init

from orca_params import connection_probability_HS19, excitatory_neurons,\
    inhibitory_neurons, excitatory_synapse_soma, excitatory_synapse_dend,\
    inhibitory_synapse_soma, inhibitory_synapse_dend, synapse_mean_weight,\
    mismatch_neuron_param, mismatch_synapse_param, mismatch_plastic_param

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
            population 'x' and 'y' should be 'x_y'.
    """

    def __init__(self,
                 name='orca_wta_',
                 connectivity_params=connection_probability_HS19,
                 exc_cells_params=excitatory_neurons,
                 inh_cells_params=inhibitory_neurons,
                 exc_soma_params=excitatory_synapse_soma,
                 exc_dend_params=excitatory_synapse_dend,
                 inh_soma_params=inhibitory_synapse_soma,
                 inh_dend_params=inhibitory_synapse_dend,
                 verbose=False,
                 num_exc_neurons=200,
                 ratio_pv=.46,
                 ratio_sst=.36,
                 ratio_vip=.18,
                 noise=False):
        """ Generates building block with specified characteristics and
                elements described by Wang et al. (2018).

        Args:
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
                               verbose)

        self._groups = {}
        add_populations(self._groups,
                        group_name=name,
                        num_exc_neurons=num_exc_neurons,
                        ratio_pv=ratio_pv,
                        ratio_sst=ratio_sst,
                        ratio_vip=ratio_vip,
                        verbose=verbose,
                        exc_cells_params=exc_cells_params,
                        inh_cells_params=inh_cells_params,
                        noise=noise
                        )
        add_connections(self._groups,
                        group_name=name,
                        verbose=verbose,
                        connectivity_params=connectivity_params,
                        exc_soma_params=exc_soma_params,
                        exc_dend_params=exc_dend_params,
                        inh_soma_params=inh_soma_params,
                        inh_dend_params=inh_dend_params,
                        noise=noise)

    def add_input(self,
                  input_group,
                  input_name,
                  targets,
                  plasticity,
                  target_type,
                  connectivity_params=connection_probability_HS19,
                  exc_params=excitatory_synapse_soma,
                  sparsity=None):
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
        temp_groups = {}
        if plasticity == 'static':
            syn_model = static_synapse_model
        elif plasticity == 'stdp':
            syn_model = stdp_synapse_model
        elif plasticity == 'reinit':
            syn_model = reinit_synapse_model

        for target in targets:
            target_name = target.split('_')[0]
            temp_groups[f'{input_name}_{target_name}'] = Connections(
                input_group, self._groups[target],
                equation_builder=syn_model(),
                method=stochastic_decay,
                name=self.name+f'{input_name}_{target_name}_conn')

        # Make connections and set params
        syn_objects = {key: val for key, val in temp_groups.items()}
        for key, val in syn_objects.items():
            val.connect(p=connectivity_params[key])
            val.set_params(exc_params)

            # If you want to have sparsity without structural plasticity,
            # just set the desired connection probability
            if sparsity is not None:
                for neu in range(self._groups[target].N):
                    ffe_zero_w = np.random.choice(input_group.N,
                                                  int(input_group.N*sparsity),
                                                  replace=False)
                    val.weight[ffe_zero_w, neu] = 0
                    val.w_plast[ffe_zero_w, neu] = 0

                #re_init_dt = 15000*ms
                #add_group_params_re_init(groups=[val],
                #                         variable='w_plast',
                #                         re_init_variable='re_init_counter',
                #                         re_init_threshold=1,
                #                         re_init_dt=re_init_dt,
                #                         dist_param=3,
                #                         scale=1,
                #                         distribution='gamma',
                #                         clip_min=0,
                #                         clip_max=15,
                #                         variable_type='int',
                #                         reference='synapse_counter')
                #add_group_params_re_init(groups=[val],
                #                         variable='weight',
                #                         re_init_variable='re_init_counter',
                #                         re_init_threshold=1,
                #                         re_init_dt=re_init_dt,
                #                         distribution='deterministic',
                #                         const_value=1,
                #                         reference='synapse_counter')
                #add_group_params_re_init(groups=[val],
                #                         variable='tausyn',
                #                         re_init_variable='re_init_counter',
                #                         re_init_threshold=1,
                #                         re_init_dt=re_init_dt,
                #                         dist_param=5.5,
                #                         scale=1,
                #                         distribution='normal',
                #                         clip_min=4,
                #                         clip_max=7,
                #                         variable_type='int',
                #                         unit='ms',
                #                         reference='synapse_counter')

        w_init_group = list(temp_groups.values())
        if target_type=='inhibitory':
            dist_param = synapse_mean_weight['inp_i']
        else:
            dist_param = synapse_mean_weight['inp_e']

        if plasticity == 'static':
            add_group_param_init(w_init_group,
                                 variable='weight',
                                 dist_param=dist_param,
                                 scale=1,
                                 distribution='gamma',
                                 clip_min=0,
                                 clip_max=15)
            for g in w_init_group:
                g.__setattr__('weight', np.array(g.weight).astype(int))
        else:
            add_group_param_init(w_init_group,
                                 variable='w_plast',
                                 dist_param=dist_param,
                                 scale=1,
                                 distribution='gamma',
                                 clip_min=0,
                                 clip_max=15)
            for g in w_init_group:
                g.__setattr__('w_plast', np.array(g.w_plast).astype(int))
                # Zero weights untouched so they can be used for reinitializations
                g.weight[np.where(g.weight!=0)[0]] = 1

        generate_mismatch(list(temp_groups.values()),
                          mismatch_synapse_param)
        if plasticity != 'static':
            generate_mismatch(list(temp_groups.values()),
                              mismatch_plastic_param)

        self._groups.update(temp_groups)

def add_populations(_groups,
                    group_name,
                    num_exc_neurons,
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
        num_exc_neurons (int, optional): Size of excitatory population.
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
    num_inh = int(num_exc_neurons/4)
    #num_inh = int(num_exc_neurons/1.6)
    num_pv = int(num_inh * ratio_pv)
    num_sst = int(num_inh * ratio_sst)
    num_vip = int(num_inh * ratio_vip)

    pyr_cells = Neurons(num_exc_neurons,
                        equation_builder=adapt_neuron_model(num_inputs=6), #TODO 4 when I fix input?
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
                       equation_builder=static_neuron_model(num_inputs=5),
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
                        equation_builder=static_neuron_model(num_inputs=4),
                        method=stochastic_decay,
                        name=group_name+'sst_cells',
                        verbose=verbose)
    vip_cells = Neurons(num_vip,
                        equation_builder=static_neuron_model(num_inputs=4), #TODO 3 when I fix input?
                        method=stochastic_decay,
                        name=group_name+'vip_cells',
                        verbose=verbose)
    if noise:
        num_noise_cells = 30
        pyr_noise_cells = PoissonGroup(num_noise_cells, rates=28*Hz)
        vip_noise_cells = PoissonGroup(num_noise_cells, rates=10*Hz)
        num_noise_cells = 10
        pv_noise_cells = PoissonGroup(num_noise_cells, rates=1*Hz)
        sst_noise_cells = PoissonGroup(num_noise_cells, rates=1*Hz)

    pyr_cells.set_params(exc_cells_params)
    pv_cells.set_params(inh_cells_params)
    sst_cells.set_params(inh_cells_params)
    vip_cells.set_params(inh_cells_params)

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
                    verbose,
                    noise):
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
        noise (bool, optional): Flag to determine if background noise is to
            be added to neurons. This is generated with a poisson process.
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

    if noise:
        noise_pyr_conn = Connections(_groups['pyr_noise_cells'],
                                     _groups['pyr_cells'],
                                     equation_builder=static_synapse_model(),
                                     method=stochastic_decay,
                                     name=group_name+'noise_pyr_conn')
        noise_pv_conn = Connections(_groups['pv_noise_cells'],
                                     _groups['pv_cells'],
                                     equation_builder=static_synapse_model(),
                                     method=stochastic_decay,
                                     name=group_name+'noise_pv_conn')
        noise_sst_conn = Connections(_groups['sst_noise_cells'],
                                     _groups['sst_cells'],
                                     equation_builder=static_synapse_model(),
                                     method=stochastic_decay,
                                     name=group_name+'noise_sst_conn')
        noise_vip_conn = Connections(_groups['vip_noise_cells'],
                                     _groups['vip_cells'],
                                     equation_builder=static_synapse_model(),
                                     method=stochastic_decay,
                                     name=group_name+'noise_vip_conn')

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
    if noise:
        temp_groups.update({'noise_pyr': noise_pyr_conn,
                            'noise_pv': noise_pv_conn,
                            'noise_sst': noise_sst_conn,
                            'noise_vip': noise_vip_conn})
    _groups.update(temp_groups)

    # Make connections
    syn_objects = {key: val for key, val in _groups.items() if 'cells' not in key}
    for key, val in syn_objects.items():
        source, target = key.split('_')[0], key.split('_')[1]
        if source==target:
            val.connect('i!=j', p=connectivity_params[key])
        elif source=='noise':
            val.connect()
        else:
            val.connect(p=connectivity_params[key])

    if noise:
        noise_pyr_conn.tausyn = 3*ms
        noise_pv_conn.tausyn = 3*ms
        noise_sst_conn.tausyn = 3*ms
        noise_vip_conn.tausyn = 3*ms
        noise_pyr_conn.weight = 4
        noise_pv_conn.weight = 4
        noise_sst_conn.weight = 4
        noise_vip_conn.weight = 4

    # Excitatory connections onto somatic compartment
    pyr_pv_conn.set_params(exc_soma_params)
    pyr_sst_conn.set_params(exc_soma_params)
    pyr_vip_conn.set_params(exc_soma_params)

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

    if i_plast == 'plastic_inh' or i_plast == 'plastic_inh0':
        pv_pyr_conn.inh_learning_rate = 0.01
        sst_pyr_conn.inh_learning_rate = 0.01
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
                             distribution='gamma',
                             clip_min=0,
                             clip_max=15)
        for g in w_init_group:
            g.__setattr__('w_plast', np.array(g.w_plast).astype(int))
    else:
        add_group_param_init(w_init_group,
                             variable='weight',
                             dist_param=synapse_mean_weight['i_e'],
                             scale=1,
                             distribution='gamma',
                             unit=-1,
                             clip_min=0,
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
                         distribution='gamma',
                         clip_min=0,
                         clip_max=15)
    for g in w_init_group:
        g.__setattr__('w_plast', np.array(g.w_plast).astype(int))
    w_init_group = [pv_pv_conn, sst_vip_conn, vip_sst_conn]#sst_pv_conn, sst_vip_conn, vip_sst_conn]
    add_group_param_init(w_init_group,
                         variable='weight',
                         dist_param=synapse_mean_weight['i_i'],
                         scale=1,
                         distribution='gamma',
                         unit=-1,
                         clip_min=0,
                         clip_max=15)
    for g in w_init_group:
        g.__setattr__('weight', np.array(g.weight).astype(int))

    w_init_group = [pyr_pv_conn, pyr_sst_conn, pyr_vip_conn]
    add_group_param_init(w_init_group,
                         variable='weight',
                         dist_param=synapse_mean_weight['e_i'],
                         scale=1,
                         distribution='gamma',
                         clip_min=0,
                         clip_max=15)
    for g in w_init_group:
        g.__setattr__('weight', np.array(g.weight).astype(int))

    w_init_group = [pyr_pyr_conn]
    pyr_pyr_conn.weight = 1
    add_group_param_init(w_init_group,
                         variable='w_plast',
                         dist_param=synapse_mean_weight['e_e'],
                         scale=1,
                         distribution='gamma',
                         clip_min=0,
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

def generate_mismatch(mismatch_group, mismatch_params):
    """ This functions adds mismatch according to provided dictionary

    Args:
        mismatch_group (list of brian2.groups): Groups that will have mismatch
            added.
        mismatch_params (dict): Mismatch parameters of all elements in the
            group.
    """
    for g in mismatch_group:
        g.add_mismatch(std_dict=mismatch_params)
        # Convert values to integer
        for key in mismatch_params.keys():
            g.__setattr__(key, np.array(g.__getattr__(key)/ms).astype(int)*ms)
