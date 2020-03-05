Advanced tutorials
==================

WTA live plot
-------------

DVS visualizer
--------------

Sequence learning
-----------------
Online Clustering of Temporal Activity (OCTA)
---------------------------------------------


The ``OCTA`` network is a ``Buildingblock`` implementation of the canonical microcircuit
found in the cortex leveraging temporal information to extract
meaning from the input data. It consists of two two-dimensional ``WTA`` networks
(compression and prediction) connected in a recurrent manner.

It is inspired by the connectivity between the different layers of the mammalian cortex:
every element in the teili implementation has a cortical
counterpart for which the connectivity and function is preserved:

* compression['n_proj'] : Layer 4
* compression['n_exc'] : Layer 2/3
* prediction['n_exc'] : Layer 5/6

Given a high dimensional input in L2/3, the network extracts in the
recurrent connections of L4 a lower dimensional representation of
temporal dependencies by learning spatio-temporal features.

.. code-block:: python

  from brian2 import ms, prefs, defaultclock
  from teili.building_blocks.octa import Octa
  from teili.models.parameters.octa_params import wta_params, octa_params,\
      mismatch_neuron_param, mismatch_synap_param
  from teili import TeiliNetwork
  from teili.models.neuron_models import OCTA_Neuron as octa_neuron
  from teili.stimuli.testbench import OCTA_Testbench
  from teili.tools.sorting import SortMatrix


  prefs.codegen.target = "numpy"
  defaultclock.dt = 0.1 * ms

  #==========Define Network=========================================
  # create the network
  Net = TeiliNetwork()
  OCTA_net = Octa(name='OCTA_net')

  #==========Define Input=========================================
  #Input into the Layer 4 block: compression['n_proj']
  testbench_stim = OCTA_Testbench()
  testbench_stim.rotating_bar(length=10, nrows=10,
                              direction='cw',
                              ts_offset=3, angle_step=10,
                              noise_probability=0.2,
                              repetitions=300,
                              debug=False)

  OCTA_net.groups['spike_gen'].set_spikes(indices=testbench_stim.indices, times=testbench_stim.times * ms)

#==========Add new block to Net=========================================

  Net.add(OCTA_net,
          OCTA_net.monitors['spikemon_proj'],
          OCTA_net.sub_blocks['compression'],
          OCTA_net.sub_blocks['prediction'])

  Net.run(np.max(testbench_stim.times) * ms,
          report='text')

To be able to visualize the compressed activity it is necessary to sort the activity in
the compressed representation.

.. code-block:: python

  from teili.tools.sorting import SortMatrix

  weights = cp.deepcopy(np.asarray(OCTA_net.sub_blocks['compression'].groups['s_exc_exc'].w_plast))
  indices = cp.deepcopy(np.asarray(OCTA_net.sub_blocks['compression'].monitors['spikemon_exc'].i))
  time = cp.deepcopy(np.asarray(OCTA_net.sub_blocks['compression'].monitors['spikemon_exc'].t))

  s = SortMatrix(nrows=49, ncols=49, matrix=weights, axis=1)
  # We use the permuted indices to sort the neuron ids
  sorted_ind = np.asarray([np.where(np.asarray(s.permutation) == int(i))[0][0] for i in indices])

  plt.figure(1)
  plt.plot(time, sorted_ind, '.r')
  plt.xlabel('Time')
  plt.ylabel('Sorted spikes')
  plt.xlim(500,700)
  plt.title('Rasterplot compression block')
  plt.show()


Three-way networks
------------------

The ``Threeway`` block is a ``BuildingBlock`` that implements a network of
three one-dimensional ``WTA`` populations A, B and C, connected to a hidden two-dimensional ``WTA`` population H.
The role of the hidden population is to encode a relation between A, B and C, which serve as inputs and\or outputs.

In this example A, B and C encode one-dimensional values in range from 0 to 1
in a relation A + B = C to each other, which is hardcoded into connectivity of
the hidden population.


To use the block instantiate it and add to the ``TeiliNetwork``

.. code-block:: python

    from brian2 import ms, prefs, defaultclock

    from teili.building_blocks.threeway import Threeway
    from teili.tools.three_way_kernels import A_plus_B_equals_C
    from teili import TeiliNetwork

    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms

    #==========Threeway building block test=========================================

    duration = 500 * ms

    #===============================================================================
    # create the network

    exampleNet = TeiliNetwork()

    TW = Threeway('TestTW',
                  hidden_layer_gen_func = A_plus_B_equals_C,
                  monitor=True)

    exampleNet.add(TW)

    #===============================================================================
    # simulation
    # set the example input values

    TW.set_A(0.4)
    TW.set_B(0.2)

    exampleNet.run(duration, report = 'text')

    #===============================================================================
    #Visualization

    TW_plot = TW.plot()

Methods ``set_A(double)``, ``set_B(double)`` and ``set_C(double)`` send population
coded values to respective populations. Here we send A=0.2, B=0.4 and activity in
population C is inferred via H, shaping in an activity bump encoding ~0.6:

.. figure:: fig/threeway_tutorial.png
    :align: center
    :width: 800px
    :height: 400px
    :figclass: align-center

    Spike raster plot of the populations A, B and C encoding the relation A = B + C.


Teili2Genn
----------

Using the already existing brian2genn_ we can generate ``GeNN`` code which can be executed on a nVidia graphics card.
Make sure to change the ``DPIsyn`` model located in ``teiliApps/equations/DPIsyn.py``. To be able to use brian2genn_ with ``TeiliNetwork``
change this line:

.. code-block:: python

   Iin{input_number}_post = I_syn * sign(weight)  : amp (summed)

to

.. code-block:: python

   Iin{input_number}_post = I_syn * (-1 * (weight<0) + 1 * (weight>0))  : amp (summed)

Also move the following lines:

.. code-block:: python

to the ``on_pre`` key, such that it looks like:

.. code-block:: python


.. attention:: If you don't change the model `GeNN` **can't** run its code generation routines as ``Subexpressions`` are not supported.

After you made the change in ``teiliApps/equations/DPIsyn.py`` you can run the ``teili2genn_tutorial.py`` located in ``teiliApps/tutorials/``.
The ``TeiliNetwork`` is the same as in ``neuron_synapse_tutorial`` but with the specific commands to use the **genn-backend**.

.. _brian2genn: https://github.com/brian-team/brian2genn
