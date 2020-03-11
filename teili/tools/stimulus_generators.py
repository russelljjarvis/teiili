# -*- coding: utf-8 -*-
# @Author: Alpha Renner
# @Date:   2018-06-05 09:12:40

"""
The idea is to generate inputs based on a function.
This avoids having to read large datafiles and makes generation of input easier.

How to use them:
See example below.

"""

import numpy as np
import matplotlib.pyplot as plt

from brian2 import device, codegen, defaultclock, NeuronGroup, Synapses, run,\
    SpikeMonitor, StateMonitor, start_scope, Network,\
    implementation, declare_types, prefs,\
    PoissonGroup
from brian2 import mV, mA, ms, ohm, uA, pA, Hz

#import teili
from teili import normal2d_density, ind2x, ind2y, Plotter2d
from teili.core.groups import TeiliGroup, Neurons


class StimulusSpikeGenerator(TeiliGroup, PoissonGroup):
    '''
    idea: add a run_regularly with dt, that dynamically changes the rates of a
    PoissonGroup or the input current of an IF neuron

    The class is the neuron or the generator and can be added to a network

    You can add equations that specify the trajectory of any pattern-variable.
    The two variables that are given by default are trajectory_x and trajectory_y
    But you can always specify the trajectory of any other variable in the trajectory_eq

    trajectory_eq can either use a TimedArray that contains a prerecorded trajectory
    or you can also set an equation that describes the trajectory

    The first 2 args of any pattern function must always be the x/y coordinates

    Any user provided distance function has to have the brian2 decorators and take the following arguments:
    i, j, nrows, ncols (i and j are 1d indices in the 2d array)

    please have a look at the example of a moving gaussian below (if __name__ == '__main__':)
    '''

    def __init__(self, nrows, ncols, dt, trajectory_eq, amplitude=1, spike_generator='poisson',
                 pattern_func=normal2d_density, name=None, **patternkwargs):

        self.nrows = nrows
        self.ncols = ncols

        #self.dt = dt
        TeiliGroup.__init__(self)

        if name is None:
            name = '_teili_stimgen*'

        if spike_generator == 'poisson':
            PoissonGroup.__init__(self, nrows * ncols,
                                  rates='activity*Hz', name=name)
        elif spike_generator == 'IFneuron':
            raise NotImplementedError
            # This is rouhgly how it could be done;
            Neurons.__init__(self, nrows * ncols, model='''dV/dt = -V/tau + activity * amp : volt
                             tau : second''', threshold='V>1', reset='V=0', name=name)
            self.tau = patternkwargs['tau']

        else:
            raise NotImplementedError

        self.add_state_variable('activity', unit=1, shared=False,
                                constant=False, changeInStandalone=False)
        self.add_state_variable('x', unit=1, shared=False,
                                constant=False, changeInStandalone=False)
        self.add_state_variable('y', unit=1, shared=False,
                                constant=False, changeInStandalone=False)
        self.add_state_variable('amplitude', unit=1, shared=True,
                                constant=False, changeInStandalone=True)

        self.amplitude = amplitude

        self.namespace.update({"nrows": nrows,
                               "ncols": ncols,
                               "ind2x": ind2x,
                               "ind2y": ind2y,
                               })

        pattern_name = pattern_func.pyfunc.__name__
        self.namespace.update({pattern_name: pattern_func})

        # the first 2 need to be x and y
        pattern_orig_arg_names = pattern_func.pyfunc._orig_arg_names[2:]
        pattern_orig_arg_units = pattern_func.pyfunc._arg_units[2:]
        pattern_orig_arg_types = pattern_func.pyfunc._arg_types[2:]
        arg_unit_dict = dict(
            zip(pattern_orig_arg_names, pattern_orig_arg_units))
        arg_type_dict = dict(
            zip(pattern_orig_arg_names, pattern_orig_arg_types))

        if not list(pattern_orig_arg_names) == list(patternkwargs.keys()):
            raise ValueError('please specify ALL pattern_func arguments as patternkwargs\n' +
                             'The following arguments are expected:  ' + ', '.join(pattern_orig_arg_names) +
                             '\nyou only specified: ' + ', '.join(patternkwargs.keys()))

        pattern_args = patternkwargs.keys()
        pattern_args_str = ', '.join(pattern_args)

        for arg in pattern_args:
            # booleans cannot be set as statevars like that
            if arg_type_dict[arg] == 'boolean':
                self.namespace.update({arg: patternkwargs[arg]})
            else:
                self.add_state_variable(arg, unit=arg_unit_dict[arg], shared=True,
                                        constant=False)

                self.__setattr__(arg, patternkwargs[arg])

        run_reg_str = trajectory_eq + '''
                           x = ind2x(i, nrows, ncols)
                           y = ind2y(i, nrows, ncols)
                           activity=amplitude*{pattern_name}(x, y, {pattern_args})
                           '''.format(pattern_name=pattern_name, pattern_args=pattern_args_str)
        self.run_regularly(run_reg_str, dt=dt)

    def plot_rates(self):
        resh_rates = np.reshape(np.asarray(self.rates),
                                (self.nrows, self.ncols))
        plt.figure()
        plt.imshow(resh_rates)


if __name__ == '__main__':

    prefs.codegen.target = 'numpy'

    from teili.tools.cpptools import activate_standalone
    #activate_standalone(directory='test_stimgen', build_on_run=True)

    nrows = 80
    ncols = 80

    # Create a moving Gaussian with increasing sigma
    # the update that happens every dt is given in the trajectory_eq
    # the center coordinates move 5 to the right and 2 upwards every dt
    # the sigma is increased by 0.1 in both directions every dt
    trajectory_eq = '''
                    mu_x = (mu_x + 5)%nrows
                    mu_y = (mu_y + 2)%nrows
                    sigma_x += 0.1
                    sigma_y += 0.1
                    '''

    stimgen = StimulusSpikeGenerator(
        nrows, ncols, dt=50 * ms, trajectory_eq=trajectory_eq, amplitude=200,
        spike_generator='poisson', pattern_func=normal2d_density,
        name="moving_gaussian_stimgen",
        mu_x=40.0, mu_y=40.0, sigma_x=1.0, sigma_y=1.0, rho=0.0, normalized=False)

    poissonspmon = SpikeMonitor(stimgen, record=True)

    net = Network()
    net.add((stimgen, poissonspmon))
    net.run(3000 * ms)

    plt.plot(poissonspmon.t, poissonspmon.i, ".")

    plotter2d = Plotter2d(poissonspmon, (nrows, ncols))
    imv = plotter2d.plot3d(plot_dt=10 * ms, filtersize=20 * ms)
    imv.show()
