****
Core
****

Network
=======

Groups
======

Tags
----

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

see here_ for more detailed explanation of how to set and get tags from ``Groups``.

Device Mismatch
---------------

Mismatch is an inherent property of analog VLSI devices due to fabrication variability [1]_. The effect of mismatch on
chip behavior can be studied, for example, with Monte Carlo simulations [2]_.
Thus, if you are simulating neuron and synapse models of neuromorphic chips, e.g. the DPI neuron (DPI)
and the DPI synapse (DPISyn), you might also want to simulate device mismatch.
To this end, the class method ``add_mismatch()`` allows you to add a Gaussian distributed mismatch [3]_ with mean equal to the current
parameter value and standard deviation set as a fraction of the current parameter value.

As an example, once neurons and synapses are created, device mismatch can be added to some selected parameters (e.g. Itau and refP for the DPI neuron)
by specifying a dictionary with parameter names as keys and standard deviation as values, as shown in the example below.

.. code-block:: python

    import numpy as np
    from brian2 import seed
    from teili.core.groups import Neurons
    from teili.models.neuron_models import DPI

    testNeurons = Neurons(100, equation_builder=DPI(num_inputs=2))

Let's assume that the estimated mismatch distribution has a standard deviation of 10% of the current value for both parameters. Then:

.. code-block:: python

    mismatch_param = {'Itau': 0.1, 'refP': 0.1}
    testNeurons.add_mismatch(mismatch_param, seed=10)

| This will change the current parameter values by drawing random values from the specified Gaussian distribution.
If you set the mismatch seed in the input parameters, the random samples will be reproducible across simulations.
| Notice that ``self.add_mismatch()`` will automatically truncate the gaussian distribution
at zero for the lower bound. This will prevent from setting neuron/synapse parameters (which
are mainly transistor currents for the DPI model) to negative values. No upper bound is specified by default.

| However, if you want to manually specify lower bound and upper bound of the mismatch
gaussian distribution, you can use the method ``_add_mismatch_param()``, as shown below.
| With old_param being the current parameter value, this will draw samples from a Gaussian distribution with the following parameters:

* **mean**: old_param
* **standard deviation**: std * old_param
- **lower bound**: lower * std * old_param + old_param
- **upper bound**: upper * std * old_param + old_param

.. code-block:: python

    import numpy as np
    from brian2 import seed
    from teili.core.groups import Neurons
    from teili.models.neuron_models import DPI

    testNeurons = Neurons(100, equation_builder=DPI(num_inputs=2))
    testNeurons._add_mismatch_param(param='Itau', std=0.1, lower=-0.2, upper = 0.2)

Notice that this option allows you to add mismatch only to one parameter at a time.

.. [1] Sheik, Sadique, Elisabetta Chicca, and Giacomo Indiveri. "Exploiting device mismatch in neuromorphic VLSI systems to implement axonal delays." Neural Networks (IJCNN), The 2012 International Joint Conference on. IEEE, 2012.

.. [2] Hung, Hector, and Vladislav Adzic. "Monte Carlo simulation of device variations and mismatch in analog integrated circuits." Proc. NCUR 2006 (2006): 1-8.

.. [3] ...

.. _here: https://teili.readthedocs.io/en/latest/scripts/Building%20Blocks.html#tags
