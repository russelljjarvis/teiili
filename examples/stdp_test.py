# -*- coding: utf-8 -*-
# @Author: mmilde
# @Date:   2018-01-16 17:57:35
# @Last Modified by:   Moritz Milde
# @Last Modified time: 2018-06-01 16:59:19

"""this file provides an example of how to use neuron and synapse models which are present
on neurmorphic chips in the context of synaptic plasticity based on precise timing of spikes.

"""
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.exporters
import numpy as np
import os


from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second, Network, prefs,\
    SpikeMonitor, StateMonitor, figure, plot, show, xlabel, ylabel,\
    seed, xlim, ylim, subplot, network_operation, TimedArray,\
    defaultclock, SpikeGeneratorGroup, asarray, pamp, set_device, device

from teili.core.groups import Neurons, Connections
from teili import NCSNetwork, activate_standalone, deactivate_standalone
from teili.models.neuron_models import DPI
from teili.models.synapse_models import DPISyn, DPIstdp
from teili.stimuli.testbench import STDP_Testbench

save_plot = False

prefs.codegen.target = "numpy"
defaultclock.dt = 50 * us
Net = NCSNetwork()

stdp = STDP_Testbench()
gPre, gPost = stdp.stimuli(isi=30)

preSTDP = Neurons(2, equation_builder=DPI(num_inputs=1),
                  name='preSTDP', verbose=True)
preSTDP.refP = 3 * ms

postSTDP = Neurons(2, equation_builder=DPI(num_inputs=2),
                   name='postSTDP', verbose=True)


SynPre = Connections(gPre, preSTDP,
                     equation_builder=DPISyn(), name='SynPre')

SynPost = Connections(gPost, postSTDP,
                      equation_builder=DPISyn(), name='SynPost')


SynSTDP = Connections(preSTDP, postSTDP,
                      equation_builder=DPIstdp(), name='SynSTDP')

# Set parameters:
preSTDP.Itau = 6 * pA
postSTDP.Itau = 6 * pA

SynPre.connect(True)
SynPre.weight = 400.
# SynPre.weight = 100.
# SynPre.Ie_tau = 2 * pA

SynPost.connect(True)
SynPost.weight = 400.

SynSTDP.connect("i==j")
SynSTDP.weight = 100.
SynSTDP.Ie_tau = 2 * pA


spikemon_pre = SpikeMonitor(preSTDP, name='spikemon_pre')
statemon_pre = StateMonitor(preSTDP, variables='Imem',
                           record=0, name='statemon_pre')
statemon_syn_pre = StateMonitor(
    SynPre, variables=['Ie_syn'], record=0, name='statemon_syn_pre')
spikemon_post = SpikeMonitor(postSTDP, name='spikemon_post')
statemon_post = StateMonitor(
    postSTDP, variables='Imem', record=0, name='statemon_post')
statemon_syn_post = StateMonitor(SynSTDP, variables=[
                                 'Ie_syn', 'w_plast', 'weight'],
                                 record=True, name='statemon_syn_post')

Net.add(gPre, gPost, preSTDP, postSTDP, SynPre, SynPost, SynSTDP, statemon_syn_pre,
        statemon_pre, statemon_post, spikemon_pre, spikemon_post, statemon_syn_post)

duration = 2000
Net.run(duration * ms)

# Visualize
app = QtGui.QApplication([])
pg.setConfigOptions(antialias=True)

win_stdp = pg.GraphicsWindow(title="STDP Unit Test")
win_stdp.resize(1920, 1080)
win_stdp.setWindowTitle("Spike Time Dependet Plasticity")
colors = [(255, 0, 0), (89, 198, 118), (0, 0, 255), (247, 0, 255),
          (0, 0, 0), (255, 128, 0), (120, 120, 120), (0, 171, 255)]
labelStyle = {'color': '#FFF', 'font-size': '12pt'}

pImemPre = win_stdp.addPlot(title="Membrane current (Imem)")
p_pre_EPSC = win_stdp.addPlot(title="Excitatory Post Synaptic Currenc (EPSC)")
win_stdp.nextRow()
pSpikes = win_stdp.addPlot(title="STDP protocol")
p_post_EPSC = win_stdp.addPlot(title="Synaptic current Ie (STDP synapse)")
win_stdp.nextRow()
pWeight1 = win_stdp.addPlot(title="Static synaptic weight")
pWeight2 = win_stdp.addPlot(title="Plasic synaptic w_plast")

# pSpikes.addLegend()

pImemPre.plot(x=np.asarray(statemon_pre.t / ms), y=np.asarray(statemon_pre.Imem[0]),
              pen=pg.mkPen(colors[6], width=2))

p_pre_EPSC.plot(x=np.asarray(statemon_syn_pre.t / ms), y=np.asarray(statemon_syn_pre.Ie_syn[0]),
           pen=pg.mkPen(colors[5], width=2))

pSpikes.plot(x=np.asarray(spikemon_pre.t / ms), y=np.asarray(spikemon_pre.i),
             pen=None, symbol='o', symbolPen=None,
             symbolSize=7, symbolBrush=(255, 255, 255),
             name='Pre synaptic neuron')
pImemPre.setXRange(0, duration, padding=0)
p_pre_EPSC.setXRange(0, duration, padding=0)
pWeight1.setXRange(0, duration, padding=0)
pWeight2.setXRange(0, duration, padding=0)
p_post_EPSC.setXRange(0, duration, padding=0)
pSpikes.setXRange(0, duration, padding=0)
pSpikes.setYRange(-0.1, 1.1, padding=0)

text1 = pg.TextItem(text='Homoeostasis', anchor=(-0.3, 0.5))
text2 = pg.TextItem(text='Weak Pot.', anchor=(-0.3, 0.5))
text3 = pg.TextItem(text='Weak Dep.', anchor=(-0.3, 0.5))
text4 = pg.TextItem(text='Strong Pot.', anchor=(-0.3, 0.5))
text5 = pg.TextItem(text='Strong Dep.', anchor=(-0.3, 0.5))
text6 = pg.TextItem(text='Homoeostasis', anchor=(-0.3, 0.5))
pSpikes.addItem(text1)
pSpikes.addItem(text2)
pSpikes.addItem(text3)
pSpikes.addItem(text4)
pSpikes.addItem(text5)
pSpikes.addItem(text6)

text1.setPos(0, 0.5)
text2.setPos(250, 0.5)
text3.setPos(550, 0.5)
text4.setPos(850, 0.5)
text5.setPos(1150, 0.5)
text6.setPos(1450, 0.5)


pSpikes.plot(x=np.asarray(spikemon_post.t / ms), y=np.asarray(spikemon_post.i),
             pen=None, symbol='s', symbolPen=None,
             symbolSize=7, symbolBrush=(255, 0, 0),
             name='Post synaptic neuron')

p_post_EPSC.plot(x=np.asarray(statemon_syn_post.t / ms), y=np.asarray(statemon_syn_post.Ie_syn[1]),
          pen=pg.mkPen(colors[3], width=2))

for i, data in enumerate(np.asarray(statemon_syn_post.weight)):
    if i == 0:
        pWeight1.plot(x=np.asarray(statemon_syn_post.t / ms), y=data,
                      pen=pg.mkPen(colors[i], width=3))
for i, data in enumerate(np.asarray(statemon_syn_post.w_plast)):
    if i == 1:
        pWeight2.plot(x=np.asarray(statemon_syn_post.t / ms), y=data,
                      pen=pg.mkPen(colors[i], width=3))

pSpikes.setLabel('left', "Neuron ID", **labelStyle)
pSpikes.setLabel('bottom', "Time (ms)", **labelStyle)
pImemPre.setLabel('left', "Pre Imem", units='A', **labelStyle)
pImemPre.setLabel('bottom', "Time (ms)", **labelStyle)
p_pre_EPSC.setLabel('left', "Synaptic current Ie", units="A", **labelStyle)
p_pre_EPSC.setLabel('bottom', "Time (ms)", **labelStyle)
p_post_EPSC.setLabel('left', "Synapic current Ie", units='A', **labelStyle)
p_post_EPSC.setLabel('bottom', "Time (ms)", **labelStyle)
pWeight1.setLabel('left', "Synpatic weight w", **labelStyle)
pWeight1.setLabel('bottom', "Time (ms)", **labelStyle)
pWeight2.setLabel('left', "Synpatic weight w_plast", **labelStyle)
pWeight2.setLabel('bottom', "Time (ms)", **labelStyle)

b = QtGui.QFont("Sans Serif", 10)
pSpikes.getAxis('bottom').tickFont = b
pSpikes.getAxis('left').tickFont = b
pImemPre.getAxis('bottom').tickFont = b
pImemPre.getAxis('left').tickFont = b
p_pre_EPSC.getAxis('bottom').tickFont = b
p_pre_EPSC.getAxis('left').tickFont = b
p_post_EPSC.getAxis('bottom').tickFont = b
p_post_EPSC.getAxis('left').tickFont = b
pWeight1.getAxis('bottom').tickFont = b
pWeight1.getAxis('left').tickFont = b
pWeight2.getAxis('bottom').tickFont = b
pWeight2.getAxis('left').tickFont = b

QtGui.QApplication.processEvents()
if save_plot:
    plot_dir = os.getcwd()
    plot_dir = os.path.join(plot_dir, '')
    exp = pg.exporters.SVGExporter(win_stdp.scene())
    exp_img = pg.exporters.ImageExporter(win_stdp.scene())
    exp.export(plot_dir + 'stdp_test.svg')
    exp_img.export(plot_dir + 'stdp_test.png')

QtGui.QApplication.instance().exec_()
