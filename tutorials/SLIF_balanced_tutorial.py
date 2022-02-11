from brian2 import PoissonGroup
from brian2 import defaultclock, prefs
from brian2 import second, Hz, ms, ohm, mA

from teili import TeiliNetwork
from orca_wta import orcaWTA

from SLIF_utils import neuron_rate, get_metrics
from orca_params import ConnectionDescriptor, PopulationDescriptor

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pickle

defaultclock.dt = 1*ms
prefs.codegen.target = "numpy"
sim_duration = 1*second
Net = TeiliNetwork()

poisson_spikes = PoissonGroup(285, rates=6*Hz)

layer = 'L4'
path = '/Users/Pablo/git/teili/'
conn_desc = ConnectionDescriptor(layer, path)
pop_desc = PopulationDescriptor(layer, path)
for conn in conn_desc.input_plast.keys():
    conn_desc.input_plast[conn] = 'static'
for conn in conn_desc.intra_plast.keys():
    conn_desc.intra_plast[conn] = 'static'
for pop in pop_desc.group_plast.keys():
    pop_desc.group_plast[pop] = 'static'
pop_desc.group_vals['n_exc'] = 3471
pop_desc.group_vals['num_inputs']['vip_cells'] = 3
bit_res = int(sys.argv[1])
pop_desc.constants['n_bits'] = bit_res
conn_desc.constants['n_bits'] = bit_res
pop_desc.filter_params()
conn_desc.filter_params()

# Change motifs generated
for conn in conn_desc.sample.keys():
    conn_desc.sample[conn] = []
for pop in pop_desc.sample.keys():
    pop_desc.sample[pop] = []
conn_desc.intra_prob = {
    'pyr_pyr': 0.1,
    'pyr_pv': 0.1,
    'pyr_sst': 0.1,
    'pyr_vip': 0.1,
    'pv_pyr': 0.1,
    'pv_pv': 0.1,
    'sst_pv': 0.1,
    'sst_pyr': 0.1,
    'sst_vip': 0.1,
    'vip_sst': 0.1}
conn_desc.input_prob = {
    'ff_pyr': 0.25,
    'ff_pv': 0.25,
    'ff_sst': 0.25,
    'ff_vip': 0.25}

max_weight = 2**(conn_desc.constants['n_bits'] - 1) - 1
wi_perc = float(sys.argv[2])
weight_vals = {
    'ff_pyr': np.floor(.42*max_weight),
    'ff_pv': np.floor(.42*max_weight),
    'ff_sst': np.floor(.42*max_weight),
    'ff_vip': np.floor(.42*max_weight),
    'pyr_pyr': np.floor(.2*max_weight),
    'pyr_pv': np.floor(.2*max_weight),
    'pyr_sst': np.floor(.2*max_weight),
    'pyr_vip': np.floor(.2*max_weight),
    'pv_pyr': np.floor(-wi_perc*max_weight),
    'pv_pv': np.floor(-wi_perc*max_weight),
    'sst_pv': np.floor(-wi_perc*max_weight),
    'sst_pyr': np.floor(-wi_perc*max_weight),
    'sst_vip': np.floor(-wi_perc*max_weight),
    'vip_sst': np.floor(-wi_perc*max_weight),
    }
for key, weight_val in weight_vals.items():
    conn_desc.base_vals[key]['weight'] = weight_val

wta = orcaWTA(layer=layer,
               conn_params=conn_desc,
               pop_params=pop_desc,
               verbose=True,
               monitor=True)
wta._groups['pyr_pyr'].delay = 0*ms
wta._groups['pyr_cells'].g_psc = 2*ohm
wta._groups['pv_cells'].g_psc = 2*ohm
wta._groups['sst_cells'].g_psc = 2*ohm
wta._groups['vip_cells'].g_psc = 2*ohm

monitor_params = {'spikemon_pyr_cells': {'group': 'pyr_cells'},
                  'spikemon_pv_cells': {'group': 'pv_cells'},
                  'spikemon_sst_cells': {'group': 'sst_cells'},
                  'statemon_pyr_cells': {'group': 'pyr_cells',
                                         'variables': ['I', 'Vm'],
                                         'record': 0,
                                         'mon_dt': 1*ms},
                  'statemon_pv_cells': {'group': 'pv_cells',
                                         'variables': ['I', 'Vm'],
                                         'record': 0,
                                         'mon_dt': 1*ms},
                  'spikemon_vip_cells': {'group': 'vip_cells'}}
wta.create_monitors(monitor_params)

# Feedforward weights elicited only a couple of spikes on 
# excitatory neurons
wta.add_input(
    poisson_spikes,
    'ff',
    ['pyr_cells', 'pv_cells', 'sst_cells', 'vip_cells'],
    syn_types=conn_desc.input_plast,
    connectivities=conn_desc.input_prob,
    conn_params=conn_desc)

rate_results, isi, cv = [], [], []
plot_pop = 'pyr'
Net.add(wta, poisson_spikes)
Net.store()
for _ in range(10):
    Net.restore()
    Net.run(sim_duration, report='stdout', report_period=100*ms)

    pop_rates = neuron_rate(wta.monitors[f'spikemon_{plot_pop}_cells'],
        kernel_len=30*ms, kernel_var=5*ms, simulation_dt=defaultclock.dt,
        smooth=True)
    pop_avg_rates = np.mean(pop_rates['smoothed'], axis=0)
    rate_results.append(np.mean(pop_avg_rates))

    temp_isi, _ = get_metrics(wta.monitors[f'spikemon_{plot_pop}_cells'])
    for i, x in enumerate(temp_isi):
        if not isi:
            isi = temp_isi
        else:
            isi[i] = np.append(isi[i], x)
with open(f'res{bit_res}_wi{wi_perc}', 'wb') as f:
    pickle.dump(rate_results, f)

pg.mkQApp()
pw = pg.PlotWidget()
pw.resize(1536, 768)
pw.show()
p1 = pw.plotItem
p1.setLabels(left='Neuron index')

# create a new ViewBox, link the right axis to its coordinate system
p2 = pg.ViewBox()
p1.showAxis('right')
p1.scene().addItem(p2)
p1.getAxis('right').linkToView(p2)
p2.setXLink(p1)
p1.getAxis('right').setLabel('Rate')

def updateViews():
    ## view has resized; update auxiliary views to match
    global p1, p2
    p2.setGeometry(p1.vb.sceneBoundingRect())
    
    p2.linkedViewChanged(p1.vb, p2.XAxis)

updateViews()
p1.vb.sigResized.connect(updateViews)

p1.plot(np.array(wta.monitors[f'spikemon_{plot_pop}_cells'].t),
        np.array(wta.monitors[f'spikemon_{plot_pop}_cells'].i),
        pen=None, symbolBrush='w', symbolSize=4, symbol='o')
p2.addItem(pg.PlotCurveItem(
    pop_rates['t'],
    pop_avg_rates, pen=pg.mkPen(color=(200, 0, 100), width=10)))

y, x = np.histogram(isi[4], bins=np.linspace(-3, 100, 10))
win = pg.GraphicsWindow()
p1 = win.addPlot(title='ISI distribution')
font = QtGui.QFont()
font.setPixelSize(18)
p1.getAxis('bottom').setStyle(tickFont=font)
p1.getAxis('left').setStyle(tickFont=font)
p1.getAxis('bottom').setStyle(tickTextOffset = 20)
p1.getAxis('left').setStyle(tickTextOffset = 20)
p1.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150))
p2 = win.addPlot(title='Coefficient of variation')
cv = [np.std(x)/np.mean(x) for x in isi]
cv_mean = np.array([np.mean(cv)])
cv_err = np.array([np.std(cv)/np.sqrt(len(cv))])
err = pg.ErrorBarItem(x=np.array([2]), y=cv_mean, height=cv_err, beam=.02)
p2.addItem(err)

if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
