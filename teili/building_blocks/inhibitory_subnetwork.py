from brian2 import core, SpikeGeneratorGroup
from brian2 import SpikeMonitor, StateMonitor
from brian2 import ms, mV, pA

from teili import BuildingBlock
from teili.models.neuron_models import ExpAdaptLIF
# TODO: Add ADP or VDP plasticity for alpha synapse
from teili.models.synapse_models import Alpha

interneuron_params = {'we_exc_inh': 1,
                      'we_ff_inh': 0.8,
                      'wi_inh_exc': -1,
                      'wi_inh_inh': -1,
                      'rp_inh': 1 * ms,
                      'ei_connection_probability': 1,
                      'ie_connection_probability': 1,
                      'ii_connection_probability': 0
                      }

class InhibitorySubnetwork(BuildingBlock):
    """ This class provides a simple `BuildingBlock` for inhibitory
    interneurons. The `BuildingBlock` is comprised of three distinct
    interneuron cell types (as described below).
    """

    def __init__(self, name='interneuron*',
                 neuron_eq_builder=ExpAdaptLIF,
                 synapse_eq_builder=Alpha,
                 num_neurons=12,
                 num_inputs=4,
                 block_params=interneuron_params,
                 reward_modulation=False,
                 monitor=True,
                 verbose=False):

        self.num_neurons = num_neurons
        self.reward_modulation = reward_modulation
        BuildingBlock.__init__(self,
                               name,
                               neuron_eq_builder,
                               synapse_eq_builder,
                               block_params,
                               verbose,
                               monitor)

        self._groups,\
            self.monitors,\
            self.standalone_params = gen_subnetwork(name,
                                                    neuron_eq_builder,
                                                    synapse_eq_builder,
                                                    num_neurons=num_neurons,
                                                    num_inputs=num_inputs,
                                                    monitor=monitor,
                                                    verbose=verbose,
                                                    **block_params)    

        self.spike_gen = self._groups['spike_gen']
        self.input_groups.update({'n_pv': self._groups['n_pv'],
                                  'n_sst': self._groups['n_sst']})
        self.output_groups.update({'n_pv': self._groups['n_pv']})
        self.hidden_groups.update({'n_vip': self._groups['n_vip']})

        if monitor:
            self.spikemon_exc = self.monitors['spikemon_exc']

def gen_subnetwork(name,
                   neuron_eq_builder,
                   synapse_eq_builder,
                   num_neurons,
                   num_inputs,
                   we_exc_inh=1,
                   we_ff_inh=0.8,
                   wi_inh_inh=-1,
                   wi_inh_exc=-1,
                   rp_inh=1 * ms,
                   ei_connection_probability=1, 
                   ie_connection_probability=1,
                   ii_connection_probability=0,
                   additional_statevars=[], 
                   monitor=True, 
                   verbose=False):
    """ Interneurons in the neocortex are comprised of three
    different cell types. Soma-projecting PV cells, dendrite-projecting
    SST cells and inhibitory-projecting VIP cells.
    Beside the obvious difference in their expression of different
    bio-markers (PV: Parvalbumin-expressing, SST: Somatostatin-expressing and 
    VIP: Vasoactive Intestinal Polypeptide-expressing interneurons), these
    interneuron cell types differ also in their axonal projections 
    (PV: soma of pyramidal cells, SST: dendrites of pyramidal cells and 
    soma of PV-expressing cells, VIP: soma of SST-expressing cells).
    But not only the target varies for each type also their source neuron
    population is cell-type specific. While PV- and SST-expressing inter-
    neurons receive majority of their inputs from local (inter-laminar) 
    pyramidal cells they also receive intra-laminar projections from 
    pyramidal cells of other layers (e.g. for L2/3 they receive local 
    L2/3 inputs but also feedforwad input from L4 as well as feedback
    input from L5). VIP-expressing interneurons on the other hand, 
    receive their inputs primarily from neurons located in the 
    Ventral Tegmental Area (VTA) which show reward-related activity.
    VIP-expressing interneurons thus inhibit SST-expressing interneurons
    and thus dis-inhibit PV-expressing interneurons allowing for state-
    dependent plasticity of the local EI circuitry.
    """

    return _groups, monitors, standalone_params