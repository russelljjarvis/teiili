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
For details of this building block have a look at OCTA_

.. note:: To be extended by Moritz Milde

.. _OCTA: https://code.ini.uzh.ch/mmilde/OCTA/blob/dev/README.md

