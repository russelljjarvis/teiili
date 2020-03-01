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
    :height: 200px
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
