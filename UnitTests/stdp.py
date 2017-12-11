from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.exporters
import numpy as np


from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second, Network, prefs,\
    SpikeMonitor, StateMonitor, figure, plot, show, xlabel, ylabel,\
    seed, xlim, ylim, subplot, network_operation, TimedArray,\
    defaultclock, SpikeGeneratorGroup, asarray, pamp, set_device, device

# from brian2 import *

#from NCSBrian2Lib.BuildingBlocks.BuildingBlock import BuildingBlock
from NCSBrian2Lib.Groups.Groups import Neurons, Connections
from NCSBrian2Lib import StandaloneNetwork, activate_standalone, deactivate_standalone
from NCSBrian2Lib.Stimuli.testbench import stdp_testbench

prefs.codegen.target = "numpy"
defaultclock.dt = 50 * us
Net = StandaloneNetwork()

stdp = stdp_testbench()
gPre, gPost = stdp.stimuli(isi=30)

pre = Neurons(2, baseUnit='current', adaptation='calciumFeedback', integrationMode='exponential', leak='leaky',
              position='none', noise='none', refractory=3 * ms, name="pre", numInputs=2)

post = Neurons(2, baseUnit='current', adaptation='calciumFeedback', integrationMode='exponential', leak='leaky',
               position='none', noise='none', refractory=3 * ms, name="post", numInputs=2)

SynPre = Connections(gPre, pre,
                     baseUnit='DPI', plasticity='nonplastic', name='SynPre')

SynPost = Connections(gPost, post,
                      baseUnit='DPI', plasticity='nonplastic', name='SynPost')

SynSTDP = Connections(pre, post,
                      baseUnit='DPI', plasticity='stdp', name='SynSTDP')

# Set parameters:
pre.Itau = 6 * pA
post.Itau = 6 * pA

SynPre.connect(True)
SynPre.weight = 80.

SynPost.connect(True)
SynPost.weight = 80.

SynSTDP.connect("i==j")
SynSTDP.weight = 10.
SynSTDP.Ie_tau = 2 * pA

# pre.Iconst = 1 * nA
# post.Iconst = 1.2 * nA

spikemonPre = SpikeMonitor(pre, name='spikemonPre')
statemonPre = StateMonitor(pre, variables='Imem', record=0, name='statemonPre')
statemonSynPre = StateMonitor(SynPre, variables=['Ie_syn'], record=0, name='statemonSynPre')
spikemonPost = SpikeMonitor(post, name='spikemonPost')
statemonPost = StateMonitor(post, variables='Imem', record=0, name='statemonPost')
statemonWeight = StateMonitor(SynSTDP, variables=['Ie_syn', 'wPlast', 'w'], record=True, name='statemonWeight')

Net.add(gPre, gPost, pre, post, SynPre, SynPost, SynSTDP, statemonSynPre,
        statemonPre, statemonPost, spikemonPre, spikemonPost, statemonWeight)

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

pImemPost.plot(x=np.asarray(statemonSynPre.t / ms), y=np.asarray(statemonSynPre.Ie_syn[0]),
               pen=pg.mkPen(colors[5], width=2))

pSpikes.plot(x=np.asarray(spikemonPre.t / ms), y=np.asarray(spikemonPre.i),
             pen=None, symbol='o', symbolPen=None,
             symbolSize=7, symbolBrush=(255, 255, 255),
             name='Pre synaptic neuron')
pImemPre.setXRange(0, duration, padding=0)
pImemPost.setXRange(0, duration, padding=0)
pWeight1.setXRange(0, duration, padding=0)
pWeight2.setXRange(0, duration, padding=0)
pSyn.setXRange(0, duration, padding=0)
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


pSpikes.plot(x=np.asarray(spikemonPost.t / ms), y=np.asarray(spikemonPost.i),
             pen=None, symbol='s', symbolPen=None,
             symbolSize=7, symbolBrush=(255, 0, 0),
             name='Post synaptic neuron')

pSyn.plot(x=np.asarray(statemonWeight.t / ms), y=np.asarray(statemonWeight.Ie_syn[1]),
          pen=pg.mkPen(colors[3], width=2))

for i, data in enumerate(np.asarray(statemonWeight.w)):
    if i == 0:
        pWeight1.plot(x=np.asarray(statemonWeight.t / ms), y=data,
                      pen=pg.mkPen(colors[i], width=3))
for i, data in enumerate(np.asarray(statemonWeight.wPlast)):
    if i == 1:
        pWeight2.plot(x=np.asarray(statemonWeight.t / ms), y=data,
                      pen=pg.mkPen(colors[i], width=3))

pSpikes.setLabel('left', "Neuron ID", **labelStyle)
pSpikes.setLabel('bottom', "Time (ms)", **labelStyle)
pImemPre.setLabel('left', "Pre Imem", units='A', **labelStyle)
pImemPre.setLabel('bottom', "Time (ms)", **labelStyle)
pImemPost.setLabel('left', "Post Imem", units="A", **labelStyle)
pImemPost.setLabel('bottom', "Time (ms)", **labelStyle)
pSyn.setLabel('left', "Synapic current Ie", units='A', **labelStyle)
pSyn.setLabel('bottom', "Time (ms)", **labelStyle)
pWeight1.setLabel('left', "Synpatic weight w", **labelStyle)
pWeight1.setLabel('bottom', "Time (ms)", **labelStyle)
pWeight2.setLabel('left', "Synpatic weight wPlast", **labelStyle)
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

QtGui.QApplication.processEvents()
plot_dir = '/home/moritz/Repositories/OCTA/plots/'
exp = pg.exporters.SVGExporter(win_stdp.scene())
exp_img = pg.exporters.ImageExporter(win_stdp.scene())
exp.export(plot_dir + 'stdp_test.svg')
exp_img.export(plot_dir + 'stdp_test.png')

QtGui.QApplication.instance().exec_()
