# -*- coding: utf-8 -*-
# @Author: mmilde
# @Date:   2018-01-11 14:48:17
# @Last Modified by:   mmilde
# @Last Modified time: 2018-01-25 16:29:42
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import scipy
from scipy.optimize import minimize
from scipy import ndimage

from brian2 import prefs, ms, pA, nA, StateMonitor, device, set_device,\
 second, msecond, defaultclock


from teili.building_blocks.wta import WTA, plotWTA
from teili.core.groups import Neurons, Connections
from teili.stimuli.testbench import WTA_Testbench
from teili import NCSNetwork
from teili.models.synapse_models import DPISyn, DPIstdp
from teili.tools.synaptic_kernel import kernel_gauss_1d

prefs.codegen.target = 'numpy'

run_as_standalone = True

if run_as_standalone:
    standaloneDir = os.path.expanduser('~/WTA_standalone')
    set_device('cpp_standalone', directory=standaloneDir, build_on_run=False)
    device.reinit()
    device.activate(directory=standaloneDir, build_on_run=False)
    prefs.devices.cpp_standalone.openmp_threads = 4


def gaussian(x,mu,sig):
    return np.exp(-((mu - x)**2) / (2 * sig**2))


def objective_function(par,data):
    mu = par[0]
    sig = par[1]
    ampl = par[2]
    gauss = ampl*gaussian(x,mu,sig)
    mse = np.mean([(h-g)**2 for h,g in zip(data,gauss)])
    return mse

num_neurons = 50
num_input_neurons = num_neurons

x = np.arange(0,num_neurons-1,1)

Net = NCSNetwork()
duration = 500#10000
testbench = WTA_Testbench()

wtaParams = {'weInpWTA': 500,
             'weWTAInh': 175,
             'wiInhWTA': -100,#-250,
             'weWTAWTA': 200,#75,
             'sigm': 1,
             'rpWTA': 3 * ms,
             'rpInh': 1 * ms,
             'EI_connection_probability' : 0.7,
             }

#wtaParams = {'weInpWTA': 1.5,
#             'weWTAInh': 1,
#             'wiInhWTA': -1,
#             'weWTAWTA': 0.5,
#             'sigm': 3,
#             'rpWTA': 3 * ms,
#             'rpInh': 1 * ms
#             }

gtestWTA = WTA(name='testWTA', dimensions=1, num_neurons=num_neurons, num_inh_neurons=40,
               num_input_neurons=num_input_neurons, num_inputs=2, block_params=wtaParams,
               spatial_kernel = "kernel_gauss_1d")

syn_in_ex = gtestWTA.Groups["synInpWTA1e"]
syn_ex_ex = gtestWTA.Groups['synWTAWTA1e']
syn_ex_ih = gtestWTA.Groups['synWTAInh1e']
syn_ih_ex = gtestWTA.Groups['synInhWTA1i']

testbench.stimuli(num_neurons=num_neurons, dimensions=1, start_time=100, end_time=duration)
testbench.background_noise(num_neurons=num_neurons, rate=10)

#gtestWTA.inputGroup.set_spikes(indices=testbench.indices, times=testbench.times * ms)
noise_syn = Connections(testbench.noise_input, gtestWTA,
                        equation_builder=DPISyn(), name="noise_syn",)
noise_syn.connect("i==j")
noise_syn.weight = 500

statemonWTAin = StateMonitor(gtestWTA.Groups['gWTAGroup'], ('Ie0', 'Ii0','Ie1', 'Ii1','Ie2', 'Ii2','Ie3', 'Ii3'), record=True,
                                       name='statemonWTAin')

Net.add(gtestWTA, testbench.noise_input, noise_syn, statemonWTAin)

Net.standaloneParams.update({'gtestWTA_Iconst' : 1*pA})

if run_as_standalone:
    Net.build()

#%%
#Net.printParams()
import time

st = time.time()
amplitudes_dict = {}
sigmas_dict={}

#num_cores = multiprocessing.cpu_count()
#num_workers = num_cores - 2
#pool = Pool(num_workers)
#res = pool.map(run_worker, parameter_combinations)
#pool.close()

for par0 in range(0,300,20):
    for par1 in range(0,500,20):

        if run_as_standalone:
            standaloneParams=OrderedDict([('duration', 0.5 * second),
                         ('stestWTA_e_latWeight', 400),#280),
                         ('stestWTA_e_latSigma', 2),
                         ('stestWTA_Inpe_weight', 300),
                         ('stestWTA_Inhe_weight', par1),#300),
                         ('stestWTA_Inhi_weight', -par0),
                         ('gtestWTA_refP', 5. * msecond),
                         ('gtestWTA_Inh_refP', 5. * msecond),
                         ('gtestWTA_Iconst', 5000 * pA)])

            duration=standaloneParams['duration']/ms
            Net.run(duration=duration*ms, standaloneParams=standaloneParams, report='text')
        else:
            Net.run(duration * ms)

        num_source_neurons = gtestWTA.Groups['gWTAInpGroup'].N
        num_target_neurons = gtestWTA.Groups['gWTAGroup'].N
        cm = plt.cm.get_cmap('jet')
        #x = np.arange(0, num_target_neurons, 1)
        #y = np.arange(0, num_source_neurons, 1)
        #X, Y = np.meshgrid(x, y)
        #data = np.zeros((num_target_neurons, num_source_neurons)) * np.nan
        # Getting sparse weights
        #wta_plot,_=plotWTA(name='testWTA', start_time=0 * ms, end_time=duration * ms,
        #        WTAMonitors=gtestWTA.Monitors, plot_states=False)


        spikemonWTA = gtestWTA.Groups['spikemonWTA']
        spiketimes = spikemonWTA.t
        dt = defaultclock.dt
        spikeinds = np.asarray(spiketimes/dt, dtype = 'int')

        data_sparse = scipy.sparse.coo_matrix((np.ones(len(spikeinds)),(spikeinds,[i for i in spikemonWTA.i])))
        data_dense = data_sparse.todense()

        #data_dense.shape
        filtersize = 500*ms
        data_filtered = ndimage.uniform_filter1d(data_dense, size=int(filtersize / dt), axis=0, mode='constant') * second / dt
        #plt.plot(data) #[400,:])
        data = data_filtered[400,:]

        from functools import partial
        minres = minimize(partial(objective_function,data=data),[10,3,50])#,method='COBYLA')

        ampl = minres.x[2]
        mu =  minres.x[0]
        sig = minres.x[1]

        gauss_fit =  ampl*gaussian(x, mu, sig)

        #plt.plot(x,gauss_fit)
        #plt.plot(x,data)
        #plt.legend(labels = ['fit','data'])
        #plt.show()


        amplitudes_dict[(par0,par1)] = ampl
        sigmas_dict[(par0,par1)] = sig


print('took', time.time()-st)


plt.plot(amplitudes_dict.keys(),amplitudes_dict.values())
plt.plot(sigmas_dict.keys(),sigmas_dict.values())


parammap = {par:amplitudes_dict[par] for par in amplitudes_dict if amplitudes_dict[par]<150}
paramnames = ["excexc", "inh"]
plot_param_map(parammap=parammap, paramnames=paramnames)


parammap = {par:sigmas_dict[par] for par in sigmas_dict if sigmas_dict[par]<10 and sigmas_dict[par]>0}
paramnames = ["excexc", "inh"]
plot_param_map(parammap=parammap, paramnames=paramnames)

plt.show()

if False:
    plt.plot(syn_ex_ex.weight)#[syn_ex_ex.i==20])
    plt.plot(syn_ex_ex.weight[20,:])
    ex_ex_sum = np.sum(syn_ex_ex.weight[20,:] * syn_ex_ex.baseweight_e[20,:])
    ex_sum = np.sum(syn_ex_ih.weight * syn_ex_ih.baseweight_e)
    ih_sum = np.sum(syn_ih_ex.weight * syn_ih_ex.baseweight_i)


    ex_ex_mat = np.zeros((100,100))
    ex_ex_mat[syn_ex_ex.i,syn_ex_ex.j] =  syn_ex_ex.weight * syn_ex_ex.baseweight_e
    plt.imshow(ex_ex_mat)

    from numpy import linalg
    w,v=linalg.eig(ex_ex_mat)
    plt.plot(w)
    plt.imshow(v)

    statemonWTA = gtestWTA.Groups['statemonWTA']

    gWTAGroup = gtestWTA.Groups["gWTAGroup"]
    gWTAGroup.print_equations()

    gWTAGroup.Ie0/pA
    gWTAGroup.Ie1/pA
    gWTAGroup.Ie2/pA
    gWTAGroup.Ie3/pA
    gWTAGroup.Ii0/pA
    gWTAGroup.Ii1/pA
    gWTAGroup.Ii2/pA
    gWTAGroup.Ii3/pA

    Ie = gWTAGroup.Ie0/pA+gWTAGroup.Ie1/pA+gWTAGroup.Ie2/pA+gWTAGroup.Ie3/pA
    Ii = gWTAGroup.Ii0/pA+gWTAGroup.Ii1/pA+gWTAGroup.Ii2/pA+gWTAGroup.Ii3/pA

    plt.figure()
    plt.plot(Ie)
    plt.plot(Ii)
    plt.plot(Ii/Ie)
    plt.show()


    statemonWTAin.Ie0/pA
    statemonWTAin.Ie1/pA
    statemonWTAin.Ie2/pA
    statemonWTAin.Ie3/pA
    statemonWTAin.Ii0/pA
    statemonWTAin.Ii1/pA
    statemonWTAin.Ii2/pA
    statemonWTAin.Ii3/pA

    Ie = statemonWTAin.Ie0/pA+statemonWTAin.Ie1/pA+statemonWTAin.Ie2/pA+statemonWTAin.Ie3/pA
    Ii = statemonWTAin.Ii0/pA+statemonWTAin.Ii1/pA+statemonWTAin.Ii2/pA+statemonWTAin.Ii3/pA

    Ie_sum = np.sum(Ie.T,axis=1)
    Ii_sum = np.sum(Ii.T,axis=1)

    plt.figure()
    plt.plot(Ie_sum)
    plt.plot(Ie_sum)
    plt.figure()
    plt.plot(Ii_sum/Ie_sum)
    plt.show()

    font = 10
    fig = plt.figure(figsize=(8,6))
    ax1 = plt.subplot(211)
    ax1.plot(statemonWTA.t / ms, statemonWTA.Imem[0] / nA)
    ax1.set_title('Step neurons input: Iconst', fontsize=font)
    ax1.set_xlabel('Time (ms)', fontsize=font - 2)
    ax1.set_ylabel('Iconst (nA)', fontsize=font - 2)
    ax1.tick_params(axis='x', labelsize=font - 4)
    ax1.tick_params(axis='y', labelsize=font - 4)

#%%
standaloneParams=OrderedDict([('duration', 0.5 * second),
             ('stestWTA_e_latWeight', 400),#280),
             ('stestWTA_e_latSigma', 2),
             ('stestWTA_Inpe_weight', 300),
             ('stestWTA_Inhe_weight', 200),#300),
             ('stestWTA_Inhi_weight', -20),

             ('gtestWTA_refP', 5. * msecond),
             ('gtestWTA_Inh_refP', 5. * msecond),
             ('gtestWTA_Iconst', 5000 * pA)])

duration=standaloneParams['duration']/ms
Net.run(duration=duration*ms, standaloneParams=standaloneParams, report='text')

wta_plot,_=plotWTA(name='testWTA', start_time=0 * ms, end_time=duration * ms,
        WTAMonitors=gtestWTA.Monitors, plot_states=False)
wta_plot.show()


spikemonWTA = gtestWTA.Groups['spikemonWTA']
spiketimes = spikemonWTA.t
dt = defaultclock.dt
spikeinds = spiketimes/dt

data_sparse = scipy.sparse.coo_matrix((np.ones(len(spikeinds)),(spikeinds,[i for i in spikemonWTA.i])))
data_dense = data_sparse.todense()

#data_dense.shape
filtersize = 500*ms
data_filtered = ndimage.uniform_filter1d(data_dense, size=int(filtersize / dt), axis=0, mode='constant') * second / dt
plt.plot(data_filtered[-10]) #[400,:])