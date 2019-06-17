# -*- coding: utf-8 -*-
"""
This is the same as the tutorial file but uses subgroups.
It is used to test issues with subroups in brian2 and the library.
Later, a subroup unit test should be created

Created on 25.8.2017
"""

from pyqtgraph.Qt import QtGui
import pyqtgraph as pg


from brian2 import ms, nA, second, prefs, SpikeMonitor, StateMonitor, SpikeGeneratorGroup, asarray

from teili.core.groups import Neurons, Connections
from teili import TeiliNetwork
from teili.models.neuron_models import DPI
from teili.models.synapse_models import DPISyn

prefs.codegen.target = "numpy"
# defaultclock.dt = 10 * us

tsInp = asarray([1, 3, 4, 5, 6, 7, 8, 9]) * ms
indInp = asarray([0, 0, 0, 0, 0, 0, 0, 0])
gInpGroup = SpikeGeneratorGroup(1, indices=indInp, times=tsInp, name='gtestInp')


Net = TeiliNetwork()

testNeurons = Neurons(2, equation_builder=DPI(num_inputs=2), name="testNeuron")
testNeurons.refP = 3 * ms

testNeurons2 = Neurons(2, equation_builder=DPI(num_inputs=2), name="testNeuron2")
testNeurons2.refP = 3 * ms

testNeuronsSub1 = testNeurons[0:1]
testNeuronsSub2 = testNeurons[1:2]

#InpSyn = Connections(gInpGroup, testNeurons, equation_builder=DPISyn(), name="testSyn", verbose=True)
#InpSyn.connect(True)
InpSyn1 = Connections(gInpGroup, testNeuronsSub1, equation_builder=DPISyn(), name="testSyn1a", verbose=True)
InpSyn1.connect(True)
InpSyn2 = Connections(gInpGroup, testNeuronsSub2, equation_builder=DPISyn(), name="testSyn1b", verbose=True)
InpSyn2.connect(True)

InpSyn1.weight = 10
InpSyn2.weight = 10

Syn = Connections(testNeurons, testNeurons2, equation_builder=DPISyn(), name="testSyn2")
Syn.connect(True)

# you can change all the parameters like this after creation of the neurongroup:
Syn.weight = 100

testNeurons.Iconst = 7 * nA
# testNeurons2.Itau = 13 * pA
# testNeurons2.Iath = 80 * pA
# testNeurons2.Iagain = 20 * pA
# testNeurons2.Ianorm = 8 * pA


spikemonInp = SpikeMonitor(gInpGroup, name='spikemonInp')
spikemon = SpikeMonitor(testNeurons, name='spikemon')
spikemonOut = SpikeMonitor(testNeurons2, name='spikemonOut')
statemonInpSyn = StateMonitor(
    InpSyn1, variables='I_syn', record=True, name='statemonInpSyn')
statemonNeuOut = StateMonitor(testNeurons2, variables=[
                              'Imem'], record=0, name='statemonNeuOut')
statemonNeuIn = StateMonitor(testNeurons, variables=[
                             "Iin", "Imem", "Iahp"], record=[0, 1], name='statemonNeu')
statemonSynOut = StateMonitor(
    Syn, variables='I_syn', record=True, name='statemonSynOut')

Net.add(gInpGroup, testNeurons, testNeurons2, InpSyn1, InpSyn2, Syn, spikemonInp, spikemon,
        spikemonOut, statemonNeuIn, statemonNeuOut, statemonSynOut, statemonInpSyn)

duration = 0.500
Net.run(duration * second)

# Visualize simulation results
from teili.tools.visualizer.DataControllers.Rasterplot import Rasterplot
from teili.tools.visualizer.DataControllers.Lineplot import Lineplot
from teili.tools.visualizer.DataViewers import PlotSettings

app = QtGui.QApplication.instance()
pg.setConfigOptions(antialias=True)
MyPlotSettings = PlotSettings(fontsize_title=12,
                              fontsize_legend=12,
                              fontsize_axis_labels=10,
                              marker_size=7)

win = pg.GraphicsWindow(title='teili Test Simulation')
win.resize(1900, 600)
win.setWindowTitle('Simple SNN')

p1 = win.addPlot()
p2 = win.addPlot()
win.nextRow()
p3 = win.addPlot()
p4 = win.addPlot()
win.nextRow()
p5 = win.addPlot()
p6 = win.addPlot()


# Spike generator
Rasterplot(MyEventsModels=[spikemonInp],
                     MyPlotSettings=MyPlotSettings,
                     time_range=[0, duration],
                     neuron_id_range=None,
                     title="Spike generator",
                     xlabel='Time (s)',
                     ylabel="Neuron ID",
                     backend='pyqtgraph',
                     mainfig=win,
                     subfig_rasterplot=p1,
                     QtApp=app,
                     show_immediately=False)

Lineplot(DataModel_to_x_and_y_attr=[(statemonInpSyn, ('t', 'I_syn'))],
                   MyPlotSettings=MyPlotSettings,
                   x_range=[0, duration],
                   title="Input synapses",
                   xlabel="Time (s)",
                   ylabel="Synaptic current (A)",
                   backend='pyqtgraph',
                   mainfig=win,
                   subfig=p2,
                   QtApp=app,
                   show_immediately=False)

Lineplot(DataModel_to_x_and_y_attr=[(statemonNeuIn, ('t', 'Imem'))],
                   MyPlotSettings=MyPlotSettings,
                   x_range=[0, duration],
                   title='Intermediate neuron',
                   xlabel="Time (s)",
                   ylabel= "Membrane current Imem (A)",
                   backend='pyqtgraph',
                   mainfig=win,
                   subfig=p3,
                   QtApp=app,
                   show_immediately=False)

Lineplot(DataModel_to_x_and_y_attr=[(statemonSynOut, ('t', 'I_syn'))],
                   MyPlotSettings=MyPlotSettings,
                   x_range=[0, duration],
                   title="Output synapses",
                   xlabel="Time (s)",
                   ylabel="Synaptic current I_e",
                   backend='pyqtgraph',
                   mainfig=win,
                   subfig=p4,
                   QtApp=app,
                   show_immediately=False)

Rasterplot(MyEventsModels=[spikemonOut],
                     MyPlotSettings=MyPlotSettings,
                     time_range=[0, duration],
                     neuron_id_range=None,
                     title="Output spikes",
                     xlabel='Time (s)',
                     ylabel="Neuron ID",
                     backend='pyqtgraph',
                     mainfig=win,
                     subfig_rasterplot=p5,
                     QtApp=app,
                     show_immediately=False)

Lineplot(DataModel_to_x_and_y_attr=[(statemonNeuOut, ('t', 'Imem'))],
                   MyPlotSettings=MyPlotSettings,
                   x_range=[0, duration],
                   title="Output membrane current",
                   xlabel="Time (s)",
                   ylabel="Membrane current I_mem",
                   backend='pyqtgraph',
                   mainfig=win,
                   subfig=p6,
                   QtApp=app,
                   show_immediately=False)

app.exec()