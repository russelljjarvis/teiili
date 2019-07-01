***************
Building Blocks
***************

The core of the motivation for the development of teili was to provide users
with a toolbox to easily build and combine neural ``BuildingBlocks`` which represent
basic algorithms implemented using neurons and synapses.
In order to provide these functionalities all ``BuildingBlocks`` share the same
parent class which amongst other provide I/O groups and properties to stack
``BuildingBlocks`` hierarchically.

BuildingBlock
=============
Each ``BuildingBlock`` has the following attributes:

.. code-block:: python

    Attributes:
            name (str, required): Name of the building_block population
            neuron_eq_builder (class, optional): neuron class as imported from
                models/neuron_models
            synapse_eq_builder (class, optional): synapse class as imported from
                models/synapse_models
            params (dictionary, optional): Dictionary containing all relevant
                parameters for each building block
            debug (bool, optional): Flag to gain additional information
            groups (dictionary): Keys to all neuron and synapse groups
            monitors (dictionary): Keys to all spike and state monitors
            monitor (bool, optional): Flag to auto-generate spike and state monitors
            standalone_params (dictionary): Dictionary for all parameters to create
                a standalone network
            sub_blocks (dictionary): Dictionary for all parent building blocks
            input_groups (dictionary): Dictionary containing all possible groups which are
                potential inputs
            output_groups (dictionary): Dictionary containing all possible groups which are
                potential outputs
            hidden (dictionary): Dictionary containing all remaining groups which are
                neither inputs nor outputs

And as each ``BuildingBlock`` inherits from this parent class all ``BuildingBlocks`` share the same attributes and properties.
To assure this every ``BuildingBlock`` initialises the ``BuildingBlock`` class:

.. code-block:: python

  BuildingBlock.__init__(self,
                         name,
                         neuron_eq_builder,
                         synapse_eq_builder,
                         block_params,
                         debug,
                         monitor)

Furthermore, as described above as soon the parent class is initialised each
building block has a set of dictionaries which handle to I/O and different ``Neuron`` and ``Connection`` groups.

The ``BuildingBlock`` class comes with a set of ``__setter__`` and ``__getter__`` functions for collecting all ``groups`` involved or identifying a subset of groups which share the same `_tags`_

To retrieve all ``Neuron``, ``Connection``, ``SpikeGeneratorGroup`` etc. simply call the ``groups`` property

.. code-block:: python

     test1DWTA = WTA(name='test1DWTA', dimensions=1, num_neurons=16, debug=False)
     bb_groups = test1DWTA.groups

Tags
======================

Each ``TeiliGroup`` has an attribute called ``_tags``. The idea behind the ``_tags`` are that the user can easily define a dictionary and use this dictionary to gather all ``TeiliGroups`` which share the same ``_tags``.

Tags should be set as the network expands and the functionality changes.

Tags are defined as:

* **mismatch**: (bool) Mismatch present of group
* **noise**: (bool) Noise input, noise connection or noise presence
* **level**: (int) Level of hierarchy in the building blocks. WTA groups are level 1, OCTA groups are level 2 etc
* **sign**: (str : exc/inh/None) Sign of neuronal population. 
* **target sign**: (str : exc/inh/None) Sign of target population. None if not applicable.
* **num_inputs**: (int) Number of inputs in Neuron population. None if not applicable.
* **bb_type**: (str : WTA/ OCTA/ 3-WAY) Building block type.
* **group_type**: (str : Neuron/Connection/ SpikeGen) Group type
* **connection_type**: (str : rec/lateral/fb/ff/None) Connection type

Setting Tags
--------------
Tags can be set:

.. code-block:: python

  test1DWTA = WTA(name='test1DWTA', dimensions=1, num_neurons=16, debug=False)
  target_group = test1DWTA._groups['n_exc']
  basic_tags_empty =   { 'mismatch' : 0,
                'noise' : 0,
                 'level': 0 ,
                  'sign': 'None',
                  'target sign': 'None',
                  'num_inputs' : 0,
                  'bb_type' : 'None',
                  'group_type' : 'None',
                  'connection_type' : 'None',
        }

  test1DWTA._set_tags(basic_tags_empty, target_group)

and updated:

.. code-block:: python

  test1DWTA._tags['mismatch'] = 1

Getting Tags
--------------------
Specific groups can filtered using tags:

.. code-block:: python

  test1DWTA.get_groups({'group_type': 'SpikeGenerator'})

All tags of a group can be obtained by:

.. code-block:: python

  test1DWTA.print_tags('n_exc')


Winner-takes-all (WTA)
======================

Every building block has a set of parameters such as weights and refractory period, which can be specified outside the building block generation and unpacked to the building block. For the WTA building_block this dictionary looks as follows:

.. code-block:: python

      wta_params = {'we_inp_exc': 1.5,
                    'we_exc_inh': 1,
                    'wi_inh_exc': -1,
                    'we_exc_exc': 0.5,
                    'sigm': 3,
                    'rp_exc': 3 * ms,
                    'rp_inh': 1 * ms,
                    'ei_connection_probability': 1,
                    'ie_connection_probability': 1,
                    'ii_connection_probability': 0}

where each key is defined as:

* **we_inp_exc**: Excitatory synaptic weight between input SpikeGenerator and WTA neurons.
* **we_exc_inh**: Excitatory synaptic weight between WTA population and inhibitory interneuron.
* **wi_inh_exc**: Inhibitory synaptic weight between inhibitory interneuron and WTA population.
* **we_exc_exc**: Self-excitatory synaptic weight (WTA).
* **sigm**: Standard deviation in number of neurons for Gaussian connectivity kernel.
* **rp_exc**: Refractory period of WTA neurons.
* **rp_inh**: Refractory period of inhibitory neurons.
* **wiInhInh**: Self-inhibitory weight of the interneuron population.
* **ei_connection_probability**: WTA to interneuron connectivity probability.
* **ie_connection_probability**: Interneuron to WTA connectivity probability
* **ii_connection_probability**: Interneuron to Interneuron connectivity probability.

Now we can import the necessary modules and build our building block.

.. code-block:: python

      from teili.building_blocks.wta import WTA
      from teili.models.neuron_models import DPI

1Dimensional WTA
----------------

The WTA building block comes in two slightly different versions. The versions only differ in the dimensionality of the WTA.

.. code-block:: python
      # The number of neurons in your WTA population.
      # Note that this number is squared in the 2D WTA
      num_neurons = 50
      # The number of neurons which project to your WTA.
      # Note that this number is squared in the 2D WTA
      num_input_neurons = 50
      my_wta = WTA(name='my_wta', dimensions=1,
                   neuron_eq_builder=DPI,
                   num_neurons=num_neurons, num_inh_neurons=int(num_neurons**2/4),
                   num_input_neurons=num_input_neurons, num_inputs=2,
                   block_params=wta_params,
                   monitor=True)

2Dimensinal WTA
---------------

To generate a 2 dimensional WTA population you can do the following.

.. code-block:: python

      # The number of neurons in your WTA population.
      # Note that this number is squared in the 2D WTA
      num_neurons = 7
      # The number of neurons which project to your WTA.
      # Note that this number is squared in the 2D WTA
      num_input_neurons = 10
      my_wta = WTA(name='my_wta', dimensions=2,
                   neuron_eq_builder=DPI,
                   num_neurons=num_neurons, num_inh_neurons=int(num_neurons**2/4),
                   num_input_neurons=num_input_neurons, num_inputs=2,
                   block_params=wta_params,
                   monitor=True)

Changing a certain synapse group from being static to plastic:

.. code-block:: python

      from teili.core.groups import Connections
      from teili.models.synapse_models import DPIstdp
      my_wta._groups['s_exc_exc'] = Connections(my_wta._groups['n_exc'],
                                                my_wta._groups['n_exc'],
                                                equation_builder=DPIstdp
                                                method='euler',
                                                name=my_wta._groups['s_exc_exc'].name)
      my_wta._groups['s_exc_exc'].connect(True)

Now we changed the standard DPI synapse for the recurrent connection within a WTA population to an All-to-All STDP-based DPI synapse. In order to initialize the plastic weight ``w_plast`` we need to do:

.. code-block:: python

      my_wta._groups['s_exc_exc'].weight = 45
      my_wta._groups['s_exc_exc'].namespace.update({'w_mean': 0.45})
      my_wta._groups['s_exc_exc'].namespace.update({'w_std': 0.35})
      # Initializing the plastic weight randomly
      my_wta._groups['s_exc_exc'].w_plast = 'w_mean + randn() * w_std'

Chain
=====

.. note:: TBA by Alpha Renner

Sequence learning
=================

.. note:: TBA by Alpha Renner

Threeway network
================

.. note:: TBA by Dmitrii Zendrikov

Online Clustering of Temporal Activity (OCTA)
=============================================

Online Clustering of Temporal Activity (OCTA) is a second generation building block:
it uses multiple WTA networks recurrently connected to create a cortex
inspired microcircuit that, leveraging the spike timing
information, enables investigations of emergent network dynamics [1]_.

.. image:: fig/OCTA_module.png

The basic OCTA module consists of a clustering (Layer2/3) and a prediction (L6) sub-module.
Given that all connections are subject to learning, the objective of one OCTA module is
to continuously adjust its parameters, e.g. synaptic weights and time constants, based
on local information to best capture the spatio-temporal statistics of its input.

Parameters for the network are stored in two dictionaries located in tools/octa_tools/octa_params.py

The WTA keys are explained above, the OCTA keys are defined as:

* **duration**: Duration of the simulation
* **revolutions**: Number of times input is presented
* **num_neurons**: Number of neurons in the compressionWTA. Keep in mind it is a 2D WTA.
* **num_input_neurons**: Number of neurons in the prediction WTA and in the starting data.
* **distribution**: (0 or 1) Distribution from which to initialize the weights. Gamma(1) or Normal(0).
* **dist_param_init**: Shape for Gamma distribution/ mean of normal distribution
* **scale_init**: Scale for Gamma distribution / std of normal distribution
* **dist_param_re_init**: Shape/mean for weight reinitialiazation in run_regular function
* **scale_re_init**: Scale/std for weight reinitialiazation in run_regular function
* **re_init_threshold**: (0 - 0.5) If the mean weight of a synapse is below or above (1- re_init_threshold) the weight is reinitialized
* **buffer_size**: Size of the buffer for the weight dependent regularization
* **buffer_size_plast**: Size of the buffer of the activity dependent regularization
* **noise_weight**: Synaptic weight of the noise generator
* **variance_th_c**: Variance threshold for the compression group. Parameter included in the ``activity`` synapse template.
* **variance_th_p**: Variance threshold for the prediction group.
* **learning_rate**: Learning rate
* **inh_learning_rate**: Inhibitory learning rate
* **decay**:  Decay parameter of the decay in the activity dependent run_regular
* **weight_decay**: Type of weight decay (temporal/event-based)
* **tau_stdp**: Time constant for stdp plasticity 


Initialization of the building block goes as follows:

.. code-block:: python

    from brian2 import ms
    from teili import TeiliNetwork
    from teili.building_blocks.octa import Octa
    from teili.tools.octa_tools.octa_param import wtaParameters, octaParameters,\
     octa_neuron


     OCTA =  Octa(name='OCTA',
                 wtaParams = wtaParameters,
                  octaParams = octaParameters,
                  neuron_eq_builder=octa_neuron,
                  num_input_neurons= octaParameters['num_input_neurons'],
                  num_neurons = octaParameters['num_neurons'],
                  stacked_inp = True,
                  noise= True,
                  monitor=True,
                  debug=False)


    Net = TeiliNetwork()
    Net.add(
                OCTA_net,
                OCTA_net.sub_blocks['predictionWTA'],
                OCTA_net.sub_blocks['compressionWTA']
              )
    Net.run(octaParameters['duration']*ms, report='text')

* **stacked_inp**: Flag to include an input to the network
* **noise**: Flag to include 10 Hz Poisson noise generator on ``n_exc`` of compressionWTA and predictionWTA
* **monitor**: Flag to return monitors of the network
* **debug**: Flag for verbose debug


.. note:: To be extended by Moritz Milde

.. _OCTA: https://code.ini.uzh.ch/mmilde/OCTA/blob/dev/README.md

..[1] Moritz Milde PhD thesis
