# -*- coding: utf-8 -*-
# @Author: Karla Burelo
# @Date:   2019-06-06 15:14:27
"""
This file contains unit tests the synapse equations.
"""
import unittest
import numpy as np
from math import e
from brian2 import  ms, pA, nA, amp, prefs, StateMonitor,SpikeGeneratorGroup
from teili.core.groups import Neurons, Connections
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
from teili import TeiliNetwork
from teili.models.neuron_models import DPI
from teili.models.synapse_models import DPISyn, DPIstdp, Alpha, Resonant
from teili.models.parameters.dpi_neuron_param import parameters as DPIparam

"""
NOTE: 

"""
class TestSynapticCurrent(unittest.TestCase):

    def test_dpi_synapse(self):
        """
        This tests the synpatic current of the DPIsyn model. The I_syn in
        this model is always positive however, when this is sent to the
        post-synaptic neuron it will change the sign of this current
        depending on the specified weight. To specify an excitatory synapse
        the weight should be positive and for an inhibitory synapse the
        weight should be negative. The test checks the Iin of the post-
        synaptic neuron shortly after the excitatory spike (at 0.5 ms) and
        again after the inhibitory spike (at 5 ms).
        """
        input_timestamps = np.asarray([0.5,5]) * ms
        input_indices = np.asarray([0,1])
        # Set spike generator
        input_spikegenerator = SpikeGeneratorGroup(2,
                                                   indices=input_indices,
                                                   times=input_timestamps,
                                                   name='gtestInp')
        Net = TeiliNetwork()
        # Set neuron group
        output_neuron = Neurons(1,
                                equation_builder=DPI(num_inputs=1),
                                name="output_neuron")

        output_neuron.set_params(DPIparam)
        # Set Connections
        input_output_synapse = Connections(input_spikegenerator, output_neuron,
                            equation_builder=DPISyn(), name="input_output_synapse", verbose=False)
        input_output_synapse.connect(True)
        input_output_synapse.weight = [800,-800]
        input_output_synapse.I_tau = 5.36e-11 * amp #synaptic time constant set to 1ms

        # Set monitors
        statemon_output_neuron = StateMonitor(output_neuron, variables=["Iin"], 
            record=True, name='statemon_output_neuron')

        Net.add(input_spikegenerator,
                output_neuron,
                input_output_synapse,
                statemon_output_neuron)
        duration = 10
        Net.run(duration * ms)
        time_range = np.where(np.logical_and((statemon_output_neuron.t/ms)>=0.5,
                                             (statemon_output_neuron.t/ms)<5))
        Iin_exc_spike = (statemon_output_neuron.Iin[0]/amp)[time_range]
        time_range = np.where((statemon_output_neuron.t/ms)>=5.2)
        Iin_inh_spike = (statemon_output_neuron.Iin[0]/amp)[time_range]

        self.assertTrue(np.all(Iin_exc_spike >= 0))
        self.assertTrue(np.all(Iin_inh_spike < 0))

    def test_dpi_stdp_synapse(self):
        """
        This tests the synaptic current of the DPIstdp model. The associated
        plastic synaptic weight should increase given a positive temporal
        correlation of pre-post pairs.
        """
        input_timestamps = np.asarray([1, 3, 5, 7, 9, 11]) * ms
        input_indices = np.asarray([0,0,0,0,0,0])
        w_plast_start = 0.3
        # Set spike generator
        input_spikegenerator = SpikeGeneratorGroup(1,
                                                   indices=input_indices,
                                                   times=input_timestamps,
                                                   name='test_inp')

        output_spikegenerator = SpikeGeneratorGroup(1,
                                                    indices=input_indices,
                                                    times=input_timestamps+1*ms,
                                                    name='test_out')
        Net = TeiliNetwork()
        # Set neuron group
        input_neuron = Neurons(1,
                               equation_builder=DPI(num_inputs=1),
                               name="input_neuron")

        output_neuron = Neurons(1,
                                equation_builder=DPI(num_inputs=2),
                                name="output_neuron")

        input_neuron.set_params(DPIparam)
        output_neuron.set_params(DPIparam)
        # Set Connections
        input_synapse = Connections(input_spikegenerator,
                                    input_neuron,
                                    equation_builder=DPISyn(),
                                    name="input_synapse",
                                    verbose=False)
        output_synapse = Connections(output_spikegenerator,
                                    output_neuron,
                                    equation_builder=DPISyn(),
                                    name="output_synapse",
                                    verbose=False)
        stdp_synapse = Connections(input_neuron,
                                   output_neuron,
                                   equation_builder=DPIstdp(),
                                   name="stdp_synapse",
                                   verbose=False)

        input_synapse.connect(True)
        output_synapse.connect(True)
        stdp_synapse.connect(True)

        input_synapse.weight = 5000
        output_synapse.weight = 5000
        stdp_synapse.weight = 200
        stdp_synapse.w_plast = w_plast_start
        stdp_synapse.dApre = 0.1
        stdp_synapse.taupre = 3 * ms
        stdp_synapse.taupost = 3 * ms

        # Set monitors
        statemon_stdp_synapse = StateMonitor(stdp_synapse,
                                             variables=["w_plast"],
                                             record=True,
                                             name='statemon_stdp_synapse')

        Net.add(input_spikegenerator,
                output_spikegenerator,
                input_neuron,
                output_neuron,
                input_synapse,
                output_synapse,
                stdp_synapse,
                statemon_stdp_synapse)

        duration = 20
        Net.run(duration * ms)

        self.assertGreater(statemon_stdp_synapse.w_plast[0, -1],
                           w_plast_start)

    def test_alpha_kernel(self):
        """
        This tests the alpha kernel model by comparing the calculated values
        from the brian2 simulation and the analytically calculated kernel
        equation when a time vector is given. The test makes sure the error
        between the two functions is below 0.001.
        """
        input_timestamps = np.asarray([1]) * ms
        input_indices = np.asarray([0])
        # Set spike generator
        input_spikegenerator = SpikeGeneratorGroup(1,
                                                   indices=input_indices,
                                                   times=input_timestamps,
                                                   name='gtestInp',
                                                   dt = 0.05*ms)
        Net = TeiliNetwork()
        # Set neuron group
        output_neuron = Neurons(1, equation_builder=DPI(num_inputs=1),
                                name="output_neuron", dt = 0.05*ms)
        output_neuron.set_params(DPIparam)
        # Set Connections
        input_synapse_alpha = Connections(input_spikegenerator,
                                          output_neuron,
                                          equation_builder=Alpha(),
                                          name="input_synapse_alpha",
                                          verbose=False,
                                          dt = 0.05*ms,
                                          method = 'rk2')
        input_synapse_alpha.connect(True)
        input_synapse_alpha.weight = [10]
        input_synapse_alpha.baseweight = 1 * pA
        input_synapse_alpha.tausyn = 1 * ms

        # Set monitors
        statemon_input_synapse_alpha = StateMonitor(input_synapse_alpha,
                                                    variables='I_syn', 
                                                    record=True, 
                                                    name='statemon_input_synapse_alpha',
                                                    dt = 0.05*ms)


        Net.add(input_spikegenerator, output_neuron, input_synapse_alpha, statemon_input_synapse_alpha)
        duration = 5
        Net.run(duration * ms)

        # Frist identify the time where the alpha function started in brian simulation, 
        # this is related to the first spike (1*ms) and the delay that brian introduces
        I_syn_change = np.where((statemon_input_synapse_alpha.I_syn[0]/amp)>0)[0][0]
        I_syn_delay = (statemon_input_synapse_alpha.t/ms)[I_syn_change]-1

        # Normalize time to calcuate equation accurately
        I_alpha_time = ((statemon_input_synapse_alpha.t/ms)[I_syn_change:])-\
        np.min(((statemon_input_synapse_alpha.t/ms)[I_syn_change:]))

        # Calculate alpha function
        I_alpha_expected = input_synapse_alpha.weight*\
        (input_synapse_alpha.baseweight/amp)*\
        (I_alpha_time)*\
        e**(-I_alpha_time)

        # Get the calculated alopha function from brian2
        I_alpha_brian = (statemon_input_synapse_alpha.I_syn[0]/amp)[I_syn_change-1:]

        # Calculate mean squared error
        mse = np.average(((I_alpha_brian[:-1]*1e12)-(I_alpha_expected*1e12))** 2, axis=0)
        # Define an error
        error = 0.001
        # Check if mean squared error is acceptable
        self.assertLess(mse,error)

    def test_resonant_kernel(self):
        """
        This tests the resonant kernel model by comparing the calculated
        values from the brian2 simulation and the analytical calculated
        kernel equation when a time vector is given. The test makes sure the
        error between the two functions is below 0.001.
        """
        input_timestamps = np.asarray([1]) * ms
        input_indices = np.asarray([0])
        # Set spike generator
        input_spikegenerator = SpikeGeneratorGroup(1,
                                                   indices=input_indices,
                                                   times=input_timestamps,
                                                   name='gtestInp',
                                                   dt = 0.05*ms)
        Net = TeiliNetwork()
        # Set neuron group
        output_neuron = Neurons(1,
                                equation_builder=DPI(num_inputs=1),
                                name="output_neuron",
                                dt = 0.05*ms)

        output_neuron.set_params(DPIparam)
        # Set Connections
        input_synapse_resonant = Connections(input_spikegenerator,
                                             output_neuron,
                                             equation_builder=Resonant(),
                                             name="input_synapse_resonant",
                                             verbose=False,
                                             dt = 0.05*ms,
                                             method='rk2')

        input_synapse_resonant.connect(True)
        input_synapse_resonant.weight = [10]
        input_synapse_resonant.baseweight = 1 * pA
        input_synapse_resonant.tausyn = 1 * ms

        # Set monitors
        statemon_input_synapse_resonant = StateMonitor(input_synapse_resonant,
                                                       variables='I_syn',
                                                       record=True, 
                                                       name='statemon_input_synapse_resonant',
                                                       dt = 0.05*ms)


        Net.add(input_spikegenerator,
                output_neuron,
                input_synapse_resonant,
                statemon_input_synapse_resonant)

        duration = 5
        Net.run(duration * ms)

        # Frist identify the time where the alpha function started in brian simulation, 
        # this is related to the first spike (1*ms) and the delay that brian introduces
        I_syn_change = np.where((statemon_input_synapse_resonant.I_syn[0]/amp)>0)[0][0]
        I_syn_delay = (statemon_input_synapse_resonant.t/ms)[I_syn_change]-1

        # Normalize time to calcuate equation accurately
        I_resonant_time = ((statemon_input_synapse_resonant.t/ms)[I_syn_change:])-\
        np.min(((statemon_input_synapse_resonant.t/ms)[I_syn_change:]))

        # Calculate resonant function
        I_resonant_expected = input_synapse_resonant.weight*\
        (input_synapse_resonant.baseweight/amp)*\
        e**(-I_resonant_time)*np.sin(3 * I_resonant_time)

        # Get the calculated alopha function from brian2
        I_resonant_brian = (statemon_input_synapse_resonant.I_syn[0]/amp)[I_syn_change-1:]

        # Calculate mean squared error
        mse = np.average(((I_resonant_brian[:-1]*1e12)-(I_resonant_expected*1e12))** 2, axis=0)
        # Define an error
        error = 0.001

        # Check if mean squared error is acceptable
        self.assertLess(mse,error)


if __name__ == '__main__':
    unittest.main(verbosity=1)
