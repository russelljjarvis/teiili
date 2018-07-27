#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a tutorial that shows how to add mismatch to a SNN. 
As an example, here we connect one input neuron to 1000 output neurons. 
(Neuron model: DPI, Synapse model: DPISyn).
The standard deviation of the mismatch distribution is specified in the 
dictionaries:
    - mismatch_neuron_param
    - mismatch_synap_param
Here mismatch is added to the neuron refractory period (refP) and to the synaptic 
weight (baseweight_e).

Created on Wed Jul 25 18:32:44 2018
@author: nrisi
"""

import pyqtgraph as pg
import numpy as np
from brian2 import SpikeGeneratorGroup, SpikeMonitor, StateMonitor, ms, asarray, nA, prefs
from teili.core.groups import Neurons, Connections
from teili import teiliNetwork
from teili.models.neuron_models import DPI as neuron_model
from teili.models.synapse_models import DPISyn as syn_model

prefs.codegen.target = "numpy"

Net = teiliNetwork()

mismatch_neuron_param = {
'Inoise' : 0,
'Iconst' : 0,
'kn' : 0,
'kp' : 0,
'Ut' : 0,
'Io' : 0,
'Cmem' : 0,
'Iath' : 0,
'Iagain' : 0,
'Ianorm' : 0,
'Ica' : 0,
'Itauahp' : 0,
'Ithahp' : 0,
'Cahp' : 0,
'Ishunt' : 0,
'Ispkthr' : 0,
'Ireset' : 0,
'Ith' : 0,
'Itau' : 0,
'refP' : 0.2,
}

mismatch_synap_param = {
'Io_syn' : 0,
'kn_syn' : 0,
'kp_syn' : 0,
'Ut_syn' : 0,
'Csyn' : 0,
'Ie_tau' : 0,
'Ii_tau' : 0,
'Ie_th' : 0,
'Ii_th' : 0,
'Ie_syn' : 0,
'Ii_syn' : 0,
'w_plast' : 0,
'baseweight_e' : 0.2,
'baseweight_i' : 0,
}

# Input layer
ts_input = asarray([1, 3, 4, 5, 6, 7, 8, 9]) * ms
ids_input = asarray([0, 0, 0, 0, 0, 0, 0, 0])
input_spikegen = SpikeGeneratorGroup(1, indices=ids_input,
                                  times=ts_input, name='gtestInp')

# Output layer
output_neurons = Neurons(1000, equation_builder=neuron_model(num_inputs=2), 
                     name='output_neurons') 
output_neurons.refP = 3 * ms
output_neurons.Iconst = 10 * nA

# Input Synapse
input_syn = Connections(input_spikegen, output_neurons, equation_builder=syn_model(), 
                     name="inSyn", verbose=False)
input_syn.connect(True)
input_syn.weight = 5

"""Adding mismatch to neurons and synaptic weights:
getattr(output_neurons, mism_param_neu) returns an array of length equal to the
number of neurons. Assuming that mismatch has not been added yet (e.g. if you
have just created the neuron population), the values of the selected parameter
will be the same for all the neurons in the population. (we choose to store 
the first one)
"""
mean_neuron_param = np.copy(getattr(output_neurons, 'refP'))[0]
unit_old_param_neu = getattr(output_neurons, 'refP').unit
mean_synapse_param = np.copy(getattr(input_syn, 'baseweight_e'))[0]
unit_old_param_syn = getattr(input_syn, 'baseweight_e').unit

output_neurons.add_mismatch(std_dict=mismatch_neuron_param, seed=10)
input_syn.add_mismatch(std_dict=mismatch_synap_param, seed=11)
#%%
# Setting monitors:
spikemon_input = SpikeMonitor(input_spikegen, name='spikemon_input')
spikemon_output = SpikeMonitor(output_neurons, name='spikemon_output')
statemon_output = StateMonitor(output_neurons,
                           variables=['Imem'],
                           record=True, 
                           name='statemonNeuMid')
statemon_input_syn = StateMonitor(input_syn, 
                           variables='Ie_syn', 
                           record=True, 
                           name='statemon_input_syn')

Net.add(input_spikegen, output_neurons, input_syn,
        spikemon_input, spikemon_output, 
        statemon_output, statemon_input_syn)

# Run simulation for 500 ms
duration = 500
Net.run(duration * ms)
              
#%% 
pg.setConfigOptions(antialias=True)
labelStyle = {'color': '#FFF', 'font-size': '12pt'}
colors = [(255, 255, 255), (255, 0, 0), (89, 198, 118), (0, 0, 255), (247, 0, 255),
          (0, 0, 0), (255, 128, 0), (120, 120, 120), (0, 171, 255)]

# Rasterplot and statemonitor
win1 = pg.GraphicsWindow(title='teili Test Simulation')
win1.resize(1900, 900)
win1.setWindowTitle('Simple SNN')
p1 = win1.addPlot(title="Spike generator")
win1.nextRow()
p2 = win1.addPlot(title="Output layer")
win1.nextRow()
p3 = win1.addPlot(title="EPSC")
win1.nextRow()
p4 = win1.addPlot(title="I_mem")

p1.plot(x=np.asarray(spikemon_input.t / ms), y=np.asarray(spikemon_input.i),
        pen=None, symbol='o', symbolPen=None,
        symbolSize=2, symbolBrush=colors[0])
p1.setLabel('left', "Neuron ID", **labelStyle)
p1.setLabel('bottom', "Time (ms)", **labelStyle)
p1.setXRange(0, duration, padding=0)

p2.plot(x=np.asarray(spikemon_output.t / ms), y=np.asarray(spikemon_output.i),
        pen=None, symbol='o', symbolPen=None,
        symbolSize=2, symbolBrush=colors[1])
p2.setLabel('left', "Neuron ID", **labelStyle)
p2.setLabel('bottom', "Time (ms)", **labelStyle)
p2.setXRange(0, duration, padding=0)

neuron_ids_to_plot = np.random.randint(1000, size=5)
for i, data in enumerate(np.asarray(statemon_input_syn.Ie_syn[neuron_ids_to_plot])):
    name = 'Syn_{}'.format(i)
    p3.plot(x=np.asarray(statemon_input_syn.t / ms), y=data,
            pen=pg.mkPen(colors[i], width=2), name=name)
p3.setLabel('left', "EPSC", units="A", **labelStyle)
p3.setLabel('bottom', "Time (ms)", **labelStyle)
p3.setXRange(0, duration, padding=0)

for i, data in enumerate(np.asarray(statemon_output.Imem[neuron_ids_to_plot])):
    p4.plot(x=np.asarray(statemon_output.t / ms), y=data,
            pen=pg.mkPen(colors[i], width=3))
p4.setLabel('left', "Membrane current Imem", units="A", **labelStyle)
p4.setLabel('bottom', "Time (ms)", **labelStyle)    
p4.setXRange(0, duration, padding=0)    

# Mismatch distribution
win2 = pg.GraphicsWindow(title='teili Test Simulation')
win2.resize(1900, 600)
win2.setWindowTitle('Mismatch distribution')
p1 = win2.addPlot(title='baseweight_e')
win2.nextRow()
p2 = win2.addPlot(title='refP')

y,x = np.histogram(np.asarray(getattr(input_syn, 'baseweight_e')), bins="auto")
curve = pg.PlotCurveItem(x=x, y=y, stepMode=True, brush=(0, 0, 255, 80))
p1.addItem(curve)
p1.plot(x=np.asarray([mean_synapse_param, mean_synapse_param]), y=np.asarray([0, np.max(y)]), 
        pen=pg.mkPen((255, 0, 0), width=2))
p1.setLabel('bottom', units=str(unit_old_param_syn), **labelStyle)

y,x = np.histogram(np.asarray(getattr(output_neurons, 'refP')), bins="auto")
curve = pg.PlotCurveItem(x=x, y=y, stepMode=True, brush=(0, 0, 255, 80))
p2.addItem(curve)
p2.plot(x=np.asarray([mean_neuron_param, mean_neuron_param]), y=np.asarray([0, np.max(y)]), 
        pen=pg.mkPen((255, 0, 0), width=2))
p2.setLabel('bottom', units=str(unit_old_param_neu), **labelStyle)