****
Core
****

Network
=======
``TeiliNetwork`` is a subclass of ``brian2.Network``. It does the same thing plus some additional methods for convenience.
There are properties to get all ``monitors``, ``Neurons`` and ``Connections`` that were added to the Network.

Like in `Brian2`, there is an ``add`` method to which all ``Groups`` have to be added, for usage, please refer to the `teili` examples (in particular `neuron_synapse_tutorial`_) and to the `Brian2 documentation`_ ().


Groups
======

Neurons
-------
``Neurons`` is a subclass of ``brian2.NeuronGroup`` and can be used in the same way.
Have a look at `neuron_synapse_tutorial`_ for an introduction.
In teili there are different ways to initialize a ``Neurons`` object:

.. code-block:: python

    import os
    from teili.core.groups import Neurons
    from teili.models.neuron_models import DPI
    from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
    # the teili way
    G = Neurons(100, equation_builder=DPI(num_inputs=2))
    # from a static file
    path = os.path.expanduser("~")
    model_path = os.path.join(path, "teiliApps", "equations", "")
    neuron_model = NeuronEquationBuilder.import_eq(
        filename=model_path + 'DPI.py', num_inputs=2)
    G = Neurons(100, equation_builder=neuron_model)
    # or the brian2 way
    G = Neurons(100, model='dv/dt = -v / tau : 1')

As in brian2_ we provide a ``Neuron`` class which inherits from brian2's ``NeuronGroup`` class.
The required keyword arguments are the same as described in brian2's `neuron tutorial`_.
See below a example use case of pre-defined ``neuron_models``.
For static ``neuron_model`` usage please refer to 
``teiliApps/tutorials/neuron_synapse_import_eq_tutorial.py``.

.. code-block:: python

    from teili import Neurons
    from teili.models.neuron_models import DPI as neuron_model

    test_neurons = Neurons(2, equation_builder=neuron_model(num_inputs=2),
                            name="test_neurons")

where **num_inputs** defines how many distinct inputs the ``NeuronGroup`` is expecting.
This allows us to potentially treat each synaptic connection as independent and not to
perform a linear summation before each current is injected into the neuron.
For many simulations this is an unnecessary feature as most models expect a linear summation
of all synaptic inputs.
By defining the number of inputs explicitly, however, one can study branch specific inputs
with a distribution of synaptic time constants which are asynchronously integrated.

Each model, whether ``Neuron`` or ``Connection`` is internally generated dynamically
using the EquationBuilder_. For more details please refer to NeuronEquationBuilder_.

An example of the ``neuron_model`` class is shown below:

.. code-block:: python

    class DPI(NeuronEquationBuilder):
        """This class provides you with all equations to simulate a current-based
        exponential, adaptive leaky integrate and fire neuron as implemented on
        the neuromorphic chips by the NCS group. The neuron model follows the DPI neuron
        which was published in 2014 (Chicca et al. 2014).
        """

        def __init__(self, num_inputs=1):
            """This initializes the NeuronEquationBuilder with DPI neuron model.

            Args:
                num_inputs (int, optional): Description
            """
            NeuronEquationBuilder.__init__(self, base_unit='current', adaptation='calcium_feedback',
                                          integration_mode='exponential', leak='leaky',
                                          position='spatial', noise='none')
            self.add_input_currents(num_inputs)

The ``NeuronEquationBuilder`` has the following keyword arguments:

* **base_unit**: Either set to ``current`` or ``voltage`` depending whether you want to simulate current-based hardware neuron models
* **adaptation**: Toggles spike-frequency adaptation mechanism in ``neuron_model``. Can either be set to ``None`` or ``calcium_feedback``.
* **integration_mode**: Can be set to ``linear``, ``quadratic`` or ``exponential``
* **leak**: Toggles leaky integration. Possible values are ``leaky`` or ``non_leaky``.
* **position**: Adds positional x, y attribute to neuron in order to spatially arrange the neurons. Once the ``neuron_model`` has these attributes the user can access and set them by ``neuron_obj.x``. 
* **noise**: Adds constant noise to ``neuron_model``

The reason behind this is that the ``EquationBuilder`` has access to a set of templates defined in ``teili/models/builder/templates/`` such that the same neuron model can easily be simulated with and without leak for example. Of course we offer the possibility of a work-around so that statically defined models can be simulated. For details please refer to the tutorial_

For more information please consult the `EquationBuilder`_ section.
Let's connect neurons to one another.

Connections
-----------
The ``Connections`` class is a subclass of ``brian2.Synapses`` and can be used in the same way.
Have a look at `neuron_synapse_tutorial`_ for an introduction.
In `teili` there are different ways to initialize a ``Connections`` object:

.. code-block:: python

    import os
    from teili.core.groups import Connections
    from teili.models.synapse_models import DPISyn
    from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
    # the teili way
    S = Connections(pre_neuron, post_neuron,
                    equation_builder=DPISyn(), name="synapse_name")
    # from a static file
    path = os.path.expanduser("~")
    model_path = os.path.join(path, "teiliApps", "equations", "")
    synapse_model = = SynapseEquationBuilder.import_eq(
        model_path + 'DPISyn.py')
    S = Connections(pre_neuron, post_neuron,
                    equation_builder=synapse_model, name="synapse_name")
    # or the brian2 way
    S = Connections(pre_neuron, post_neuron, model='w : volt', on_pre='v += w')
    
As in brian2_ we provide a ``Connections`` class which inherits from brian2's ``Synapses`` class.
The required keyword arguments are the same as described in brian2's `synapse tutorial`_.
See below a example use case of pre-defined ``synapse_models``.
For static ``synapse_model`` usage please refer to 
``~/teiliApps/tutorials/neuron_synapse_builderobj_tutorial.py``.

.. code-block:: python

  from teili.core.groups import Neurons, Connections
  from teili.models.synapse_models import DPISyn as syn_model

  test_synapse = Connections(test_neurons1, test_neurons2,
                             equation_builder=syn_model(),
                             name="test_synapse")



Each model, whether ``Neuron`` or ``Connection`` is internally generated dynamically
using the EquationBuilder_. For more details please refer to NeuronEquationBuilder_ or SynapseEquationBuilder_

An example of the ``synapse_model`` class is shown below:

.. code-block:: python

  class DPISyn(SynapseEquationBuilder):
      """This class provides you with all the equations to simulate a Differential Pair
      Integrator (DPI) synapse as published in Chicca et al. 2014.
      """

      def __init__(self):
          """This class provides you with all the equations to simulate a Differential Pair
          Integrator (DPI) synapse as published in Chicca et al. 2014.
          """
          SynapseEquationBuilder.__init__(self, base_unit='DPI',
                                          plasticity='non_plastic')

The ``SynapseEquationBuilder`` has the following keyword arguments:

* **base_unit**: Set to ``current`` or ``conductance`` depending whether you want to simulate current-based hardware neuron models. This keyword argument can also be set to ``DPI`` or ``DPIShunting`` for specific hardware model simulation.
* **kernel**: Can be set to ``exponential``, ``alpha`` or ``resonant`` which ultimately sets the shapes of the EPSC and IPSC.
* **plasticity**: This keyword argument lets you easily generate any ``synapse_model`` with either an ``stdp`` or ``fusi`` learning rule.

The reason behind this is that the ``EquationBuilder`` has access to a set of templates defined in ``teili/models/builder/templates/`` such that the same ``synapse_model`` can easily be simulated with and without plasticity or with different plasticity rules for example.
Of course we offer the possibility of a work-around so that statically defined models can be simulated.
For details please refer to the `plasticity tutorial`_

.. note:: TBA Contributing guide for new templates


Tags
====

Each ``TeiliGroup`` has an attribute called ``_tags``. For more information please see here_ for more detailed explanation of how to set and get tags from ``Groups``.

Tags should be set as the network expands and the functionality changes.
Tags are defined as:

* **mismatch**: (bool) Flag to indicate if mismatch is present in the ``Group``
* **noise**: (bool) Noise input, noise connection or noise presence
* **level**: (int) Level of BuildingBlock in the hierarchy. A WTA BuildingBlock which is connected directly to a sensor array is level 1. An OCTA BuildinBlock, however, is level 2 as it consists of level 1 WTAs
* **sign**: (str : exc/inh/None) Sign on neuronal population. Following Dale law.
* **target_sign**: (str : exc/inh/None) Sign of target population. None if not applicable.
* **num_inputs**: (int) Number of inputs in Neuron population. None if not applicable.
* **bb_type**: (str : WTA/ OCTA/ 3-WAY) Building block type.
* **group_type**: (str : Neuron/Connection/ SpikeGen) Group type
* **connection_type**: (str : rec/lateral/fb/ff/None) Connection type

Setting Tags
------------
Tags can be set:
.. code-block:: python

  test_wta._set_tags({'custom_tag' : custom_tag }}, target_group)


Getting Tags
------------
Specific groups can be filtered using specific tags:

.. code-block:: python

  test_wta.get_groups({'group_type': 'SpikeGenerator'})

All tags of a group can be obtained by:

.. code-block:: python

  test_wta.print_tags('n_exc')


Device Mismatch
===============

Mismatch is an inherent property of analog VLSI devices due to fabrication variability [1]_. The effect of mismatch on chip behavior can be studied, for example, with Monte Carlo simulations [2]_.
Thus, if you are simulating neuron and synapse models of neuromorphic chips, e.g. the DPI neuron (DPI) and the DPI synapse (DPISyn), you might also want to simulate device mismatch.
To this end, the class method ``add_mismatch()`` allows you to add a Gaussian distributed mismatch with mean equal to the current parameter value and standard deviation set as a fraction of the current parameter value.

As an example, once ``Neurons`` and ``Connections`` are created, device mismatch can be added to some selected parameters (e.g. `Itau` and `refP` for the `DPI neuron`) by specifying a dictionary with parameter names as ``keys`` and standard deviation as ``values``, as shown in the example below.
If no dictionary is passed to ``add_mismatch()`` 20% mismatch will be added to all variables except for variables that are found in ``teili/models/parameters/no_mismatch_parameter.py``.

.. code-block:: python

    import numpy as np
    from brian2 import seed
    from teili.core.groups import Neurons
    from teili.models.neuron_models import DPI

    test_neurons = Neurons(100, equation_builder=DPI(num_inputs=2))

Let's assume that the estimated mismatch distribution has a standard deviation of 10% of the current value for both parameters. Then:

.. code-block:: python

    mismatch_param = {'Itau': 0.1, 'refP': 0.1}
    test_neurons.add_mismatch(mismatch_param, seed=10)

This will change the current parameter values by drawing random values from the specified Gaussian distribution.

If you set the mismatch seed in the input parameters, the random samples will be reproducible across simulations.

.. note:: Note that ``self.add_mismatch()`` will automatically truncate the Gaussian distribution

at zero for the lower bound. This will prevent neuron or synapse parameters (which
are mainly transistor currents for the DPI model) from being set to negative values. No upper bound is specified by default.
However, if you want to manually specify the lower bound and upper bound of the mismatch
Gaussian distribution, you can use the method ``_add_mismatch_param()``, as shown below.
With old_param being the current parameter value, this will draw samples from a Gaussian distribution with the following parameters:

* **mean**: old_param
* **standard deviation**: std * old_param
* **lower bound**: lower * std * old_param + old_param
* **upper bound**: upper * std * old_param + old_param

.. code-block:: python

    import numpy as np
    from brian2 import seed
    from teili.core.groups import Neurons
    from teili.models.neuron_models import DPI

    test_neurons = Neurons(100, equation_builder=DPI(num_inputs=2))
    test_neurons._add_mismatch_param(param='Itau', std=0.1, lower=-0.2, upper = 0.2)

.. note:: that this option allows you to add mismatch only to one parameter at a time.

.. [1] Sheik, Sadique, Elisabetta Chicca, and Giacomo Indiveri. "Exploiting device mismatch in neuromorphic VLSI systems to implement axonal delays." Neural Networks (IJCNN), The 2012 International Joint Conference on. IEEE, 2012.
.. [2] Hung, Hector, and Vladislav Adzic. "Monte Carlo simulation of device variations and mismatch in analog integrated circuits." Proc. NCUR 2006 (2006): 1-8.

.. _here: https://teili.readthedocs.io/en/latest/scripts/Building%20Blocks.html#tags
.. _neuron_synapse_tutorial: https://teili.readthedocs.io/en/latest/scripts/Tutorials.html#neuron-synapse-tutorial
.. _Brian2 documentation: https://brian2.readthedocs.io/en/stable/user/running.html#networks
.. _tutorial: https://teili.readthedocs.io/en/latest/scripts/Tutorials.html#import-equation-from-a-file
.. _plasticity tutorial: https://teili.readthedocs.io/en/latest/scripts/Tutorials.html#stdp-tutorial
.. _neuron tutorial: https://brian2.readthedocs.io/en/stable/resources/tutorials/1-intro-to-brian-neurons.html
.. _synapse tutorial: https://brian2.readthedocs.io/en/stable/resources/tutorials/2-intro-to-brian-synapses.html
.. _brian2: https://brian2.readthedocs.io/en/stable/index.html
.. _EquationBuilder: https://teili.readthedocs.io/en/latest/scripts/Equation%20builder.html#
.. _NeuronEquationBuilder: https://teili.readthedocs.io/en/latest/modules/teili.models.builder.html#module-teili.models.builder.neuron_equation_builder
.. _SynapseEquationBuilder: https://teili.readthedocs.io/en/latest/modules/teili.models.builder.html#module-teili.models.builder.synapse_equation_builder

