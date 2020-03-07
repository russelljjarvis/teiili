****************
Equation Builder
****************

The equation builder serves as a dynamic model generator. It takes model templates, located in ``teili/models/builder/templates/`` and combines these template snippets using ``teili/models/builder/combine.py``.

There are two distinct equation builder classes:

* ``NeuronEquationBuilder``
* ``SynapseEquationBuilder``

Each builder is wrapped by a neuron/synapse model generator class located in ``teili/models/``:

* ``neuron_model``
* ``synapse_model``

Keyword arguments for builder
=============================
In order to generate a neuron/synapse model, its builder needs to be initialized by specifying a ``base_unit`` and a set of values which will define the model itself and thus which template equation/parameters are combined.

The values that determine the model should be passed by defining a keyword which explains the functionality.

NeuronEquationBuilder keywords
------------------------------

.. code-block:: python

    from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
    num_inputs = 2
    my_neuron_model = NeuronEquationBuilder.__init__(base_unit='current',
                                                     adaptation='calcium_feedback',
                                                     integration_mode='exponential',
                                                     leak='leaky',
                                                     position='spatial',
                                                     noise = 'None')
    my_neuron.add_input_currents(num_inputs)


The keywords used in the example and the values are explained below:

* **base_unit**: Indicates whether the neuron model is ``current`` or ``voltage`` based.
* **adaptation**: Determines what type of adaptive feedback should be used. Can be ``calciumfeedback`` or ``None``.
* **integration_mode**: Determines how the neuron integrates up to spike-generation. Can be ``linear`` or ``exponential``.
* **leak**: Enables leaky integration. Can be ``leaky`` or ``non_leaky``.
* **position**: To enable spatial-like position indices (x, y) about the position of a neuron in space. Can be ``spatial`` or ``None``.
* **noise**: Determines what type of distribution is used to inject noise into the neuron. Can be ``gaussian`` or ``None``.

Custom keywords (such as gain_modulation or activity_modulation) can be added by defining a custom equation template in ``teili/models/builder/templates/neuron_templates.py`` and adding the keyword to either the ``current_equation_sets`` or to the ``voltage_equation_sets`` dictionary.
When defining a new neuron model, import the new feature by passing the newly constructed keyword to the ``NeuronEquationBuilder``.

SynapseEquationBuilder keywords
-------------------------------

.. code-block:: python

    from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
    my_synapse_model = SynapseEquationBuilder.__init__(base_unit='DPI',
                                                       plasticity='non_plastic')

The keywords used in the example and the values are explained below:

* **base_unit**: Indicates whether synapse uses ``current``, ``conductance`` or ``DPI`` current models.
* **kernel**: Specifies temporal kernel with which each spike gets convolved. Can be ``exponential``, ``resonant`` or ``alpha``.
* **plasticity**: Plasticity algorithm for the synaptic weight. Can either be ``non_plastic``, ``fusi`` or
  ``stdp``.

Custom keywords (such as new learning rules or new kernels) can be added by defining a custom equation template in ``teili/models/builder/templates/synapse_templates.py`` and adding the keywords to the ``synaptic_equations`` dictionary.
When defining a new synapse model, import the new feature by passing the newly constructed keyword to the ``SynapseEquationBuilder``.

Equations that do not fit into the existing synaptic modes:  *current*, **conductance**, **DPI**, **DPI shunting** can be grouped into the **unit_less** mode and the equation needs to be added to the **unit_less_parameters** dictionary.

Dictionary structure
====================

Both ``EquationBuilders`` a dictionary attribute, the keys of which represent the keywords necessary to generate a neuron or synapse model in order to simulate it using `brian2`.
The keywords given to the EquationBuilder class are used to select template dictionaries which are combined.
This is done by passing these keywords to ``current_equation_sets`` and ``current_parameters`` in the case of neurons and to ``modes``, ``kernels``, ``plasticity_models`` and ``current_parameters``
in the case of synapses.

.. code-block:: python

    # In the case of neurons
    keywords = combine_neu_dict(eq_templ, param_templ)
    # In the case of synapses
    keywords = combine_syn_dict(eq_tmpl, param_templ)


Neuron model keywords
---------------------

The dictionary ``keywords`` has the following keys:

.. code-block:: python

    keywords = {'model': keywords['model'],
                'threshold': keywords['threshold'],
                'reset': keywords['reset'],
                'refractory': 'refP',
                'parameters': keywords['parameters']}

Synapse model keywords
----------------------

The dictionary ``keywords`` has the following keys:

.. code-block:: python

    keywords = {'model': keywords['model'],
                'on_pre': keywords['on_pre'],
                'on_post': keywords['on_post'],
                'parameters': keywords['parameters']}

Class methods
=============

import_eq
---------

A function to import pre-defined neuron_model. This function can load a dictionary and its keywords in order to initialize the ``EquationBuilder``.

.. code-block:: python

    from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder

    my_neu_model = NeuronEquationBuilder.import_eq(
        '~/teiliApps/equations/DPI', num_inputs=2)

where ``num_inputs`` specifies how many distinct neuron populations project to the target population.

For synapses the import works as follows:

.. code-block:: python

    from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder

    my_syn_model = SynapseEquationBuilder.import_eq(
        'teiliApps/equations/DPISyn')

export_eq
---------

In order to generate models which can later be changed manually and imported again, the ``EquationBuilder`` class features an export method which can be used as follows:

.. code-block:: python

    path = '/home/YOU/teiliApps/equations/'
    DPI = NeuronEquationBuilder(base_unit='current', adaptation='calcium_feedback',
                                integration_mode='exponential', leak='leaky',
                                position='spatial', noise='none')
    DPI.add_input_currents(num_inputs)
    DPI.export_eq(os.path.join(path, "DPI"))

For synapse models:

.. code-block:: python

    path = '/home/YOU/teiliApps/equations/`)
    dpi_syn = SynapseEquationBuilder(base_unit='DPI',
                                   plasticity='non_plastic')

    dpi_syn.export_eq(os.path.join(path, "DPISyn"))

.. note:: The path can be any existing path. You do not need to store your models within the teiliApps directory.

var_replacer
------------

This function takes two equation sets in the form of strings and replaces all lines which start with '%'.

.. code-block:: python

    '%x = theta' --> 'x = theta'
    '%x' --> ''

This feature allows equations that we don't want to compute to be removed from the template by writing '%[variable]' in the other equation blocks.

To replace variables and lines:

.. code-block:: python

    from teili.models.builder.combine import var_replacer
    var_replacer(first_eq, second_eq, params)
