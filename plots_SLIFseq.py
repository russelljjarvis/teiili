import numpy as np
import copy

from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
from pyqtgraph.dockarea import *

from pathlib import Path

from teili.tools.sorting import SortMatrix

trials = []
for i in Path('/home/pablo/data_low_number_neurons/').glob('data_3_50_0.85_2_2_-3.*'):
    trials.append(i)

app = QtGui.QApplication([])
win = QtGui.QMainWindow()
area = DockArea()
win.setCentralWidget(area)

docks = []
for trial in trials:
    docks.append(Dock(trial.name[4:-4], size=(1, 1)))
for dock in docks:
    area.addDock(dock, 'left')
for i in range(len(docks) - 1):
    area.moveDock(docks[i-1], 'above', docks[i-2])

for i, trial in enumerate(trials):
    num_inh = trial.name[:-4].split('_')[1]
    input_rate = trial.name[:-4].split('_')[2]
    ei_conn = trial.name[:-4].split('_')[3]
    ffi_weight = trial.name[:-4].split('_')[4]
    ei_weight = trial.name[:-4].split('_')[5]
    ie_weight = trial.name[:-4].split('_')[6]
    num_exc = 15
    num_channels = 30
    data=np.load(trial, allow_pickle=True)

    # Avoid storing too much data on memory
    data_length = -10000
    input_t = data['input_t'][data_length:]
    input_i = data['input_i'][data_length:]
    neuron_id_e = np.random.randint(0, num_exc)
    neuron_id_i = np.random.randint(0, num_inh)
    Vm_e = data['Vm_e'][neuron_id_e][data_length:]
    Vm_i = data['Vm_i'][neuron_id_i][data_length:]
    exc_spikes_t = data['exc_spikes_t'][data_length:]
    exc_spikes_i = data['exc_spikes_i'][data_length:]
    inh_spikes_t = data['inh_spikes_t'][data_length:]
    inh_spikes_i = data['inh_spikes_i'][data_length:]
    exc_rate_t = data['exc_rate_t'][data_length:]
    exc_rate = data['exc_rate'][data_length:]
    inh_rate_t = data['inh_rate_t'][data_length:]
    inh_rate = data['inh_rate'][data_length:]
    rf = data['rf']
    am = data['am'][:, data_length:]
    rec_ids = data['rec_ids']
    rec_w = data['rec_w']
    del data

    l = pg.LayoutWidget()
    text = f"""Num. inh. neuron: {num_inh}, Input Poisson rate: {input_rate}, e->i conn.: {ei_conn} feedforward inh. w.: {ffi_weight}, e->i w.: {ei_weight}, i->e w.: {ie_weight}"""
    l.addLabel(text)
    docks[i].addWidget(l, 0, 0, colspan=2)

    p1 = pg.PlotWidget(title='Input')
    p1.plot(input_t*1e-3, input_i, pen=None, symbolSize=3, symbol='o')
    p1.setLabel('bottom', 'Time', units='s')
    p2 = pg.PlotWidget(title='Membrane potential')
    p2.addLegend(offset=(30, 1))
    p2.plot(Vm_e, pen='r', name=f'exc. id {neuron_id_e}')
    p2.plot(Vm_i, pen='b', name=f'inh. id {neuron_id_i}')
    p2.setYRange(0, 0.025)
    p2.setLabel('left', 'Membrane potential', units='V')
    p2.setLabel('bottom', 'Time', units='s')
    docks[i].addWidget(p1, 1, 0)
    docks[i].addWidget(p2, 1, 1)

    p3 = pg.PlotWidget(title='Raster plot (exc. pop.)')
    p3.plot(exc_spikes_t*1e-3, exc_spikes_i, pen=None, symbolSize=3,
            symbol='o')
    p3.setLabel('left', 'Neuron index')
    p3.setLabel('bottom', 'Time', units='s')
    p3.setXLink(p1)
    p4 = pg.PlotWidget(title='Raster plot (inh. pop.)')
    p4.plot(inh_spikes_t*1e-3, inh_spikes_i, pen=None, symbolSize=3,
            symbol='o')
    p4.setLabel('left', 'Neuron index')
    p4.setLabel('bottom', 'Time', units='s')
    p4.setXLink(p1)
    docks[i].addWidget(p3, 2, 0)
    docks[i].addWidget(p4, 2, 1)

    p5 = pg.PlotWidget(title='Population rate (exc.)')
    p5.plot(exc_rate_t*1e-3,
            exc_rate,
            pen='r')
    p5.setLabel('bottom', 'Time', units='s')
    p5.setLabel('left', 'Rate', units='Hz')
    p5.setXLink(p1)
    p6 = pg.PlotWidget(title='Population rate (inh.)')
    p6.plot(inh_rate_t*1e-3,
            inh_rate,
            pen='b')
    p6.setLabel('bottom', 'Time', units='s')
    p6.setLabel('left', 'Rate', units='Hz')
    p6.setXLink(p1)
    docks[i].addWidget(p5, 3, 0)
    docks[i].addWidget(p6, 3, 1)

    # Plot matrices
    colors = [
        (0, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 255, 255)
    ]
    cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 4), color=colors)
    sorted_w = SortMatrix(ncols=num_exc, nrows=num_exc, matrix=rec_w,
            fill_ids=rec_ids) #TODO change axis
    sorted_i = np.asarray([np.where(
                    np.asarray(sorted_w.permutation) == int(i))[0][0] for i in exc_spikes_i])

    image_axis = pg.PlotItem()
    image_axis.setLabel(axis='bottom', text='RF pixels')
    image_axis.hideAxis('left')
    m1 = pg.ImageView(view=image_axis)
    m1.ui.histogram.hide()
    m1.ui.roiBtn.hide()
    m1.ui.menuBtn.hide()
    m1.setImage(np.reshape(rf, (num_channels, num_exc, -1)), axes={'t':2, 'y':0, 'x':1})
    m1.setColorMap(cmap)
    image_axis = pg.PlotItem()
    image_axis.setLabel(axis='bottom', text='RF pixels')#TODO same as below
    image_axis.hideAxis('left')
    m2 = pg.ImageView(view=image_axis)
    m2.ui.histogram.hide()
    m2.ui.roiBtn.hide()
    m2.ui.menuBtn.hide() 
    m2.setImage(np.reshape(rf, (num_channels, num_exc, -1)), axes={'t':2, 'y':0, 'x':1}) # TODO sorted rf[-1]
    m2.setColorMap(cmap)
    image_axis = pg.PlotItem()
    image_axis.setLabel(axis='bottom', text='sorted')
    image_axis.hideAxis('left')
    m3 = pg.ImageView(view=image_axis)
    m3.ui.histogram.hide()
    m3.ui.roiBtn.hide()
    m3.ui.menuBtn.hide() 
    m3.setImage(np.reshape(rf, (num_channels, num_exc, -1))[:, sorted_w.permutation,-1], axes={'y':0, 'x':1})
    m3.setColorMap(cmap)
    docks[i].addWidget(m1, 1, 3)
    docks[i].addWidget(m2, 2, 3)
    docks[i].addWidget(m3, 3, 3)
win.show()
# TODO those are global and for all t, should adapt
#spikemon.count/duration) and/or
# _ = hist(spikemon.t/ms, 100, histtype='stepfilled', facecolor='k', weights=list(ones(len(spikemon))/(N*defaultclock.dt)))
