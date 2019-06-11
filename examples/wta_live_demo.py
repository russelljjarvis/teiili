# -*- coding: utf-8 -*-
# @Author: mmilde
# @Date:   2018-01-11 14:48:17
# @Last Modified by:   mmilde
# @Last Modified time: 2018-01-25 16:29:42
import os
import numpy as np
import sys
from pyqtgraph.Qt import QtGui
from brian2 import prefs, ms, pA, nA, StateMonitor

from teili.building_blocks.wta import WTA
from teili.core.groups import Neurons, Connections
from teili.stimuli.testbench import WTA_Testbench
from teili import TeiliNetwork
from teili.models.synapse_models import DPISyn, DPIstdp
from teili.tools.synaptic_kernel import kernel_gauss_1d
from teili.tools.live import ParameterGUI, PlotGUI

from teili import NeuronEquationBuilder, SynapseEquationBuilder

DPI = NeuronEquationBuilder.import_eq('DPI', num_inputs=1)

app = QtGui.QApplication.instance()
if app is None:
    app = QtGui.QApplication(sys.argv)
else:
    print('QApplication instance already exists: %s' % str(app))

prefs.codegen.target = 'numpy'

run_as_standalone = True

num_neurons = 50
num_input_neurons = num_neurons

x = np.arange(0, num_neurons - 1, 1)

Net = TeiliNetwork()
duration = 1500  # 10000
testbench = WTA_Testbench()

wtaParams = {'weInpWTA': 500,
             'weWTAInh': 175,
             'wiInhWTA': -100,  # -250,
             'weWTAWTA': 200,  # 75,
             'sigm': 1,
             'rpWTA': 3 * ms,
             'rpInh': 1 * ms,
             'EI_connection_probability': 0.7,
             }

gtestWTA = WTA(name='testWTA', neuron_eq_builder=DPI, synapse_eq_builder=DPISyn,
               dimensions=1, num_neurons=num_neurons, num_inh_neurons=40,
               num_input_neurons=num_input_neurons, num_inputs=2, block_params=wtaParams,
               spatial_kernel="kernel_gauss_1d")

syn_in_ex = gtestWTA.Groups["synInpWTA1e"]
syn_ex_ex = gtestWTA.Groups['synWTAWTA1e']
syn_ex_ih = gtestWTA.Groups['synWTAInh1e']
syn_ih_ex = gtestWTA.Groups['synInhWTA1i']

testbench.stimuli(num_neurons=num_neurons, dimensions=1, start_time=100, end_time=duration)
testbench.background_noise(num_neurons=num_neurons, rate=10)

gtestWTA.inputGroup.set_spikes(indices=testbench.indices, times=testbench.times * ms)
noise_syn = Connections(testbench.noise_input, gtestWTA,
                        equation_builder=DPISyn(), name="noise_syn", )
noise_syn.connect("i==j")
noise_syn.weight = 500
statemonWTAin = StateMonitor(gtestWTA.Groups['gWTAGroup'],
                             ('Iin0', 'Iin1', 'Iin2', 'Iin3'), record=True,
                             name='statemonWTAin')

# We have to add shared (scalar) parameters that are influencing the other parameters if we want to change them via the GUI
# gtestWTA.Groups['gWTAGroup'].add_state_variable('', unit=1, shared=True, constant=False, changeInStandalone=False)
syn_in_ex.add_state_variable('guiweight', unit=1, shared=True, constant=False, changeInStandalone=False)
syn_in_ex.guiweight = 400
syn_in_ex.run_regularly("weight = guiweight", dt=1 * ms)

noise_syn.add_state_variable('guiweight', unit=1, shared=True, constant=False, changeInStandalone=False)
noise_syn.guiweight = 400
noise_syn.run_regularly("weight = guiweight", dt=1 * ms)

Net.add(gtestWTA, testbench.noise_input, noise_syn, statemonWTAin)

# %%
duration = 2000 * ms

plot_gui = PlotGUI(data=gtestWTA.group.Imem)

param_gui = ParameterGUI(net=Net)
#param_gui.params
param_gui.add_params(parameters=[syn_in_ex.guiweight, noise_syn.guiweight])
param_gui.showGUI()

# %%
Net.run_as_thread(duration=duration)

# syn_in_ex.guiweight
# syn_in_ex.weight

app.exec_()
