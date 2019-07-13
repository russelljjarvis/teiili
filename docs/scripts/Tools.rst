*****
Tools
*****

This is a collection of useful tools and functions mainly for use with `Brian2`. Or other spiking network software or hardware.


converter
---------
Functions in this module convert data from or to `Brian2` compatible formats.
In particular, there are functions to convert and process data from DVS cameras.

cpptools
--------
This provides functions that are used by the TeiliNetwork class to allow running the network several times with different parameters without recompiling.
This is in particular useful if you have small networks and you would like to do parameter optimization.
Have a look at `sequence_learning_standalone_tutorial`_ as an example for how to use it.

.. code-block:: python

    # net is built once
    net.build(standalone_params = standaloneParams)
    # and can then be run several times with different parameters without recompilation
    net.run(duration,standaloneParams = standaloneParams)



distance
--------
These functions to compute different distance measures are mainly there to provide an easy to use cpp implementation for `Brian2` cpp code generation.
They are used by the synaptic kernel functions, but can also be added into any `Brian2` string.
Make sure to add the functions you use to the namespace of the respective groups as follows:

.. code-block:: python

    from teili.tools.distance import function
    group.namespace['functionname_used_in_string'] = function

indexing
--------
Functions that convert 1d indices to x, y coordinates and vice versa including cpp implementation.
As `Brian2` uses 1d indexing for neurons, it is necessary to convert 1d to 2d indices every so often when e.g. generating synapses.
Numpy provides a good API for that, which we use here, but we also add a cpp implementation so the functions can be used in standalone mode.

live
----
WIP
Live plotting and changing of parameters during numpy based simulation could be done like this.

misc
----
Functions that didn't fit in any of the other categories so far.


plotter2d
---------
Will be deprecated soon once implemented into the visualizer.
Provides 2d plotting functionality (video plot and gif generation)

plotting
--------
Will be deprecated soon once implemented into the visualizer?

prob_distributions
------------------
Probability density functions with cpp implementation.

This is e.g. a plot of a 1d Gaussian:

.. code-block:: python

    import matplotlib.pyplot as plt
    dx = 0.1
    normal1drange = np.arange(-10, 10, dx)
    gaussian = [normal1d_density(x, 0, 1, True) for x in normal1drange]
    print(np.sum(gaussian) * dx)

    plt.figure()
    plt.plot(normal1drange, gaussian)
    plt.show()


random_sampling
---------------
This module provides functions to sample from a random distribution, e.g. for random initialization of weights.
All functions in should be callable in a similar way as rand() or randn() that are provided by `Brian2`.

Here is an example how this works in standalone mode:

.. code-block:: python

    from teili.tools.random_sampling import Rand_gamma, Randn_trunc

    n_samples = 10000
    standaloneDir = os.path.expanduser('~/gamma_standalone')
    set_device('cpp_standalone', directory=standaloneDir, build_on_run=True)

    ng = NeuronGroup(n_samples, '''
    testvar : 1
    testvar2 : 1''', name = 'ng_test')

    ng.namespace.update({'rand_gamma': Rand_gamma(4.60, -10750.0),
                         'randn_trunc': Randn_trunc(-1.5,1.5)
                        })

    ng.testvar = 'rand_gamma()'
    ng.testvar2 = '5*randn_trunc()'

    run(10 * ms)

    plt.figure()
    plt.title('rand_gamma')
    plt.hist(ng.testvar, 50, histtype='step')
    plt.show()

    plt.figure()
    plt.title('randn_trunc')
    plt.hist(ng.testvar2, 50, histtype='step')
    plt.show()


random_walk
-----------
Functions that generate a random walk. E.g. as artificial input.

sorting
-------
To understand the structure of in the rasterplots but also in the learned weight matrices, we need to sort the weight matrices according to some similarity measure, such as euclidean distance.
However, the sorting algorithm is completely agnostic to the similarity measure. It connects each node with maximum two edges and constructs a directed graph.
This is similar to the travelling salesman problem.

Example:
    In order to use this class you need to initialize it
    either without a filename:

.. code-block:: python

    from teili.tools.sorting import SortMatrix
    import numpy as np
    matrix = np.random.randint((49, 49))
    obj = SortMatrix(nrows=49, matrix=matrix)
    print(obj.matrix)
    print(obj.permutation)
    print(ob.sorted_matrix)

    or instead of using a matrix you can also specify a
    path to a stored matrix:

    filename = '/path/to/your/matrix.npy'
    obj = SortMatrix(nrows=49, filename=filename)

stimulus_generators
-------------------
The idea is to generate inputs based on a function instead of having to use a fixed spikegenerator that is filled before the simulation.
This avoids having to read large datafiles and makes generation of input easier.

Use it as follows (also teili groups and network can be used):

.. code-block:: python

    import matplotlib.pyplot as plt

    from brian2 import SpikeMonitor, Network, prefs, ms

    from teili.tools.stimulus_generators import StimulusSpikeGenerator
    from teili import normal2d_density, Plotter2d

    prefs.codegen.target = 'numpy'

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


synaptic_kernel
---------------
This module provides functions that can be used for synaptic connectivity kernels (generate weight matrices).
E.g. Gaussian, Mexican hat, Gabor with different dimensionality, also using different distance metrics.
In order to also use them with C++ code generation, all functions have a cpp implementation given by the @implementation decorator.


.. _sequence_learning_standalone_tutorial: https://teili.readthedocs.io/en/latest/scripts/Other%examples.html#sequence&learning
