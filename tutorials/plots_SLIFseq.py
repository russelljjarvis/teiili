import numpy as np
import copy
import pickle
import pprint

from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
from pyqtgraph.dockarea import *

import sys

from teili.tools.sorting import SortMatrix
from SLIF_utils import load_merge_multiple

sort_type = sys.argv[2]

app = QtGui.QApplication([])
win = QtGui.QMainWindow()
area = DockArea()
win.setCentralWidget(area)

d1 = Dock('traces and raster', size=(1, 1))
d2 = Dock('matrices', size=(1, 1))
d3 = Dock('receptive fields', size=(1, 1))
d4 = Dock('inh. receptive fields', size=(1, 1))
area.addDock(d1, 'left')
area.addDock(d2, 'left')
area.addDock(d3, 'left')
area.addDock(d4, 'left')
area.moveDock(d3, 'above', d4)
area.moveDock(d2, 'above', d3)
area.moveDock(d1, 'above', d2)

# Load metadata of given simulation
data_folder = sys.argv[1]
with open(f'{data_folder}metadata', 'rb') as f:
    metadata = pickle.load(f)
num_exc = metadata['num_exc']
num_pv = metadata['num_pv']
num_channels = metadata['num_channels']
selected_cells = metadata['selected_cells']
rasters = load_merge_multiple(data_folder, 'rasters*', mode='numpy')
traces = load_merge_multiple(data_folder, 'traces*', mode='numpy')
matrices = load_merge_multiple(data_folder, 'matrices*', mode='numpy',
    allow_pickle=True)
plot_d1, plot_d2, plot_d3, plot_d4 = True, True, True, True

# Avoid storing too much data on memory
input_t = rasters['input_t']
input_i = rasters['input_i']
Iin0 = traces['Iin0'][0]
Iin1 = traces['Iin1'][0]
Iin2 = traces['Iin2'][0]
Iin3 = traces['Iin3'][0]
exc_spikes_t = rasters['exc_spikes_t']
exc_spikes_i = rasters['exc_spikes_i']
inh_spikes_t = rasters['inh_spikes_t']
inh_spikes_i = rasters['inh_spikes_i']
exc_rate_t = traces['exc_rate_t']
exc_rate = traces['exc_rate']
inh_rate_t = traces['inh_rate_t']
inh_rate = traces['inh_rate']
rf = matrices['rf']
rfi = matrices['rfi']
rec_ids = matrices['rec_ids']
rec_w = matrices['rec_w']
del matrices
del rasters
del traces

if plot_d1:
    l = pg.LayoutWidget()
    text = f"""{metadata}"""
    l.addLabel(text)
    d1.addWidget(l, 0, 0, colspan=2)

    p1 = pg.PlotWidget(title='Input sequence')
    p1.plot(input_t*1e-3, input_i, pen=None, symbolSize=3, symbol='o')
    p1.setLabel('bottom', 'Time', units='s')
    p1.setLabel('left', 'Input channels')

    p2 = pg.PlotWidget(title='Membrane potential')
    p2.addLegend(offset=(30, 1))
    p2.plot(np.array(range(np.shape(Iin0)[0]))*1e-3, Iin0, pen='r', name='Iin0')
    p2.plot(np.array(range(np.shape(Iin1)[0]))*1e-3, Iin1, pen='b', name='Iin1')
    p2.plot(np.array(range(np.shape(Iin2)[0]))*1e-3, Iin2, pen='w', name='Iin2')
    p2.plot(np.array(range(np.shape(Iin3)[0]))*1e-3, Iin3, pen='g', name='Iin3')
    #p2.setYRange(0, 0.025)
    p2.setLabel('left', 'Membrane potential', units='A')
    p2.setLabel('bottom', 'Time', units='s')
    p2.setXLink(p1)

    d1.addWidget(p1, 1, 0)
    d1.addWidget(p2, 1, 1)

    # Prepate matrices
    rf_matrix = np.reshape(rf, (num_channels, num_exc, -1))[:,:,-1]
    sorted_rf = SortMatrix(ncols=num_exc, nrows=num_channels,
                           matrix=rf_matrix, axis=1,
                           similarity_metric='euclidean')
    # recurrent connections are not present in some simulations
    try:
        sorted_rec = SortMatrix(ncols=num_exc, nrows=num_exc, matrix=rec_w,
                                target_indices=rec_ids, rec_matrix=True,
                                similarity_metric='euclidean')
    except:
        sorted_rec = SortMatrix(ncols=num_exc, nrows=num_exc,
                                matrix=np.zeros((num_exc, num_exc)))

    if sort_type == 'rec_sort':
        permutation = sorted_rec.permutation
    elif sort_type == 'rate_sort':
        permutation_file = np.load(f'{data_folder}permutation.npz')
        permutation = permutation_file['ids']
    elif sort_type == 'rf_sort':
        permutation = sorted_rf.permutation
    elif sort_type == 'no_sort':
        permutation = [x for x in range(num_exc)]
    print(f'permutation indices: {permutation}')
    sorted_i = np.asarray([np.where(
                    np.asarray(permutation) == int(i))[0][0] for i in exc_spikes_i])
    p3 = pg.PlotWidget(title='Sorted raster plot (exc. pop.)')
    p3.plot(exc_spikes_t*1e-3, sorted_i, pen=None, symbolSize=3,
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
    d1.addWidget(p3, 2, 0)
    d1.addWidget(p4, 2, 1)

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
    d1.addWidget(p5, 3, 0)
    d1.addWidget(p6, 3, 1)

# Plot matrices
# Inferno colormap
colors = [
    (0, 0, 4),
    (40, 11, 84),
    (101, 21, 110),
    (159, 42, 99),
    (212, 72, 66),
    (245, 125, 21),
    (250, 193, 39),
    (252, 255, 16)
]
cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 8), color=colors)
if plot_d2:
    image_axis = pg.PlotItem()
    image_axis.setLabel(axis='bottom', text='Neuron index')
    image_axis.setLabel(axis='left', text='Input channels')
    #image_axis.hideAxis('left')
    image_axis = pg.PlotItem()
    image_axis.setLabel(axis='bottom', text='postsynaptic neuron')
    image_axis.setLabel(axis='left', text='presynaptic neuron')
    #image_axis.hideAxis('left')
    m2 = pg.ImageView(view=image_axis)
    #m2.ui.histogram.hide()
    m2.ui.roiBtn.hide()
    m2.ui.menuBtn.hide() 
    m2.setImage(sorted_rec.matrix[:, permutation][permutation, :], axes={'y':0, 'x':1})
    m2.setColorMap(cmap)
    image_axis = pg.PlotItem()
    image_axis.setLabel(axis='bottom', text='sorted RF.')
    image_axis.setLabel(axis='left', text='Input channel')
    #image_axis.hideAxis('left')
    m3 = pg.ImageView(view=image_axis)
    #m3.ui.histogram.hide()
    m3.ui.roiBtn.hide()
    m3.ui.menuBtn.hide() 
    
    m3.setImage(sorted_rf.matrix[:, permutation], axes={'y':0, 'x':1})
    m3.setColorMap(cmap)
    #m4 = pg.PlotWidget(title='Population rate')
    #m4.plot(exc_rate_t*1e-3,
    #        exc_rate,
    #        pen='r')
    #m4.plot(inh_rate_t*1e-3,
    #        inh_rate,
    #        pen='b')
    #m4.setLabel('bottom', 'Time', units='s')
    #m4.setLabel('left', 'Rate', units='Hz')
    d2.addWidget(m2, 0, 0)
    d2.addWidget(m3, 0, 1)
    #d2.addWidget(m4, 1, colspan=3)

# Plot receptive fields for each neuron
def play_receptive_fields(receptive_fields):
    for i in receptive_fields:
        i.play(1000)
if plot_d3:
    last_frame = np.reshape(rf, (num_channels, num_exc, -1))
    dims = np.sqrt(num_channels).astype(int)
    rfe = []
    j = 0
    k = 0
    for i in permutation:
        rfe.append(pg.ImageView())
        rfe[-1].ui.histogram.hide()
        rfe[-1].ui.roiBtn.hide()
        rfe[-1].ui.menuBtn.hide() 
        rfe[-1].setImage(np.reshape(last_frame[:, i, :], (dims, dims, -1)), axes={'t':2, 'y':0, 'x':1})
        rfe[-1].setColorMap(cmap)

        d3.addWidget(rfe[-1], j, k)
        if j < np.sqrt(num_exc)-1:
            j += 1
        else:
            j = 0
            k += 1

    btn = QtGui.QPushButton("play")
    btn.clicked.connect(lambda: play_receptive_fields(rfe))
    d3.addWidget(btn, j, k)
if plot_d4:
    last_frame = np.reshape(rfi, (num_channels, num_pv, -1))
    dims = np.sqrt(num_channels).astype(int)
    rfi = []
    j = 0
    k = 0
    for i in range(num_pv):
        rfi.append(pg.ImageView())
        rfi[-1].ui.histogram.hide()
        rfi[-1].ui.roiBtn.hide()
        rfi[-1].ui.menuBtn.hide() 
        rfi[-1].setImage(np.reshape(last_frame[:, i, :], (dims, dims, -1)), axes={'t':2, 'y':0, 'x':1})
        rfi[-1].setColorMap(cmap)

        d4.addWidget(rfi[-1], j, k)
        if j < np.sqrt(num_pv)-1:
            j += 1
        else:
            j = 0
            k += 1

    btn = QtGui.QPushButton("play")
    btn.clicked.connect(lambda: play_receptive_fields(rfi))
    d4.addWidget(btn, j, k)

win.show()
QtGui.QApplication.instance().exec_()
# Generate plots with matplotlib
#from brian2 import *
#figure()
#imshow(sorted_rec.matrix[:, permutation_file['ids']][permutation_file['ids'], :])
#xlabel('postsynaptic neuron')
#ylabel('presynaptic neuron')
#colorbar()
#figure()
#imshow(sorted_rf.matrix[:, permutation_file['ids']], origin='lower')
#xlabel('Neuron index')
#ylabel('Input channel')
#colorbar()
#show()
np.savez('seq.npz', seq_i=input_i, seq_t=input_t, e_raster_i=sorted_i,
         e_raster_t=exc_spikes_t, ff_matrix=sorted_rf.matrix[:, permutation],
         rec_matrix=sorted_rec.matrix[:, permutation][permutation, :])
