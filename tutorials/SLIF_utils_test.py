from brian2 import *
from tutorials.SLIF_utils import neuron_rate

def test_rates():
    pg = PoissonGroup(2, 20*Hz)
    mon = SpikeMonitor(pg)

    run(2*second)

    nr = neuron_rate(mon, kernel_len=100*ms, kernel_var=1,
                     simulation_dt=defaultclock.dt,
                     smooth=True, trials=2)
                     #interval=[1000*ms, 2000*ms], smooth=True, trials=2)

    figure()
    plot(mon.t, mon.i, '.')

    step(nr['t'], nr['rate'][0], where='pre')
    plot(nr['t'], nr['smoothed'][0])
    show()
