from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np


from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second, Network, prefs,\
    SpikeMonitor, StateMonitor, figure, plot, show, xlabel, ylabel,\
    seed, xlim, ylim, subplot, network_operation, TimedArray,\
    defaultclock, SpikeGeneratorGroup, asarray, pamp, set_device, device

from brian2 import *

#from NCSBrian2Lib.BuildingBlocks.BuildingBlock import BuildingBlock
from NCSBrian2Lib.Groups.Groups import Neurons, Connections
from NCSBrian2Lib import StandaloneNetwork, activate_standalone, deactivate_standalone
from NCSBrian2Lib.Stimuli.testbench import stdp_testbench

prefs.codegen.target = "numpy"
defaultclock.dt = 10 * us
Net = StandaloneNetwork()

stdp = stdp_testbench()
gPre, gPost = stdp.stimuli()

pre = Neurons(2, mode='current', adaptation='adaptive', transfer='exponential', leak='leaky',
              position='none', noise='none', refractory=3 * ms, name="pre", numInputs=2)

post = Neurons(2, mode='current', adaptation='adaptive', transfer='exponential', leak='leaky',
               position='none', noise='none', refractory=3 * ms, name="post", numInputs=2)

SynPre = Connections(gPre, pre,
                     kernel='exponential', plasticity='nonplastic', name='SynPre')

SynPost = Connections(gPost, post,
                      kernel='exponential', plasticity='nonplastic', name='SynPost')

SynSTDP = Connections(pre, post, mode='current',
                      kernel='exponential', plasticity='stdp', name='SynSTDP')

SynPre.connect(True)
SynPre.weight = 2

SynPost.connect(True)
SynPost.weight = 2

SynSTDP.connect("i==j")
SynSTDP.weight = 0.2

# pre.Iconst = 1 * nA
# post.Iconst = 1.2 * nA

spikemonPre = SpikeMonitor(pre, name='spikemonPre')
statemonPre = StateMonitor(pre, variables='Imem', record=0, name='statemonPre')
spikemonPost = SpikeMonitor(post, name='spikemonPost')
statemonPost = StateMonitor(post, variables='Imem', record=0, name='statemonPost')
statemonWeight = StateMonitor(SynSTDP, variables=['Ie_syn', 'wPlast'], record=True, name='statemonWeight')

Net.add(gPre, gPost, pre, post, SynPre, SynPost, SynSTDP,
        statemonPre, statemonPost, spikemonPre, spikemonPost, statemonWeight)

Net.run(500 * ms)

# Visualize
app = QtGui.QApplication([])
pg.setConfigOptions(antialias=True)

win_stdp = pg.GraphicsWindow(title="STDP Unit Test")
win_stdp.resize(1920, 1080)
win_stdp.setWindowTitle("Spike Time Dependet Plasticity")
colors = [(255, 0, 0), (89, 198, 118), (0, 0, 255), (247, 0, 255),
          (0, 0, 0), (255, 128, 0), (120, 120, 120), (0, 171, 255)]
labelStyle = {'color': '#FFF', 'font-size': '12pt'}

pImemPre = win_stdp.addPlot(title="Pre synaptic activity (Imem)")
pImemPost = win_stdp.addPlot(title="Post synaptic activity (Imem)")
win_stdp.nextRow()
pSpikes = win_stdp.addPlot(title="Pre-Post spiking activity")
pSyn = win_stdp.addPlot(title="Synaptic current Ie")
win_stdp.nextRow()
pWeight1 = win_stdp.addPlot(title="Synaptic weight")
pWeight2 = win_stdp.addPlot(title="Synaptic weight")

pSpikes.addLegend()

pImemPre.plot(x=np.asarray(statemonPre.t / ms), y=np.asarray(statemonPre.Imem[0]),
              pen=pg.mkPen(colors[6], width=2))

pImemPost.plot(x=np.asarray(statemonPost.t / ms), y=np.asarray(statemonPost.Imem[0]),
               pen=pg.mkPen(colors[5], width=2))

pSpikes.plot(x=np.asarray(spikemonPre.t / ms), y=np.asarray(spikemonPre.i),
             pen=None, symbol='o', symbolPen=None,
             symbolSize=7, symbolBrush=(255, 255, 255),
             name='Pre synaptic neuron')

pSpikes.plot(x=np.asarray(spikemonPost.t / ms), y=np.asarray(spikemonPost.i),
             pen=None, symbol='s', symbolPen=None,
             symbolSize=7, symbolBrush=(255, 0, 0),
             name='Post synaptic neuron')

pSyn.plot(x=np.asarray(statemonWeight.t / ms), y=np.asarray(statemonWeight.Ie_syn[1]),
          pen=pg.mkPen(colors[3], width=2))

for i, data in enumerate(np.asarray(statemonWeight.wPlast)):
    if i == 0:
        pWeight1.plot(x=np.asarray(statemonWeight.t / ms), y=data,
                      pen=pg.mkPen(colors[i], width=3))
    if i == 1:
        pWeight2.plot(x=np.asarray(statemonWeight.t / ms), y=data,
                      pen=pg.mkPen(colors[i], width=3))

pSpikes.setLabel('left', "Neuron ID", **labelStyle)
pSpikes.setLabel('bottom', "Time (ms)", **labelStyle)
pImemPre.setLabel('left', "Pre Imem", units='A', **labelStyle)
pImemPre.setLabel('bottom', "Time (ms)", **labelStyle)
pImemPost.setLabel('left', "Post Imem", units="A", **labelStyle)
pImemPost.setLabel('bottom', "Time", units="s", **labelStyle)
pSyn.setLabel('left', "Synapic current Ie", units='A', **labelStyle)
pSyn.setLabel('bottom', "Time (ms)", **labelStyle)
pWeight1.setLabel('left', "Synpatic weight", **labelStyle)
pWeight1.setLabel('bottom', "Time (ms)", **labelStyle)
pWeight2.setLabel('left', "Synpatic weight", **labelStyle)
pWeight2.setLabel('bottom', "Time (ms)", **labelStyle)

b = QtGui.QFont("Sans Serif", 10)
pSpikes.getAxis('bottom').tickFont = b
pSpikes.getAxis('left').tickFont = b
pImemPre.getAxis('bottom').tickFont = b
pImemPre.getAxis('left').tickFont = b
pImemPost.getAxis('bottom').tickFont = b
pImemPost.getAxis('left').tickFont = b
pSyn.getAxis('bottom').tickFont = b
pSyn.getAxis('left').tickFont = b
pWeight1.getAxis('bottom').tickFont = b
pWeight1.getAxis('left').tickFont = b
pWeight2.getAxis('bottom').tickFont = b
pWeight2.getAxis('left').tickFont = b
QtGui.QApplication.instance().exec_()
