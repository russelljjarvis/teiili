Step-by-step guide to creating new equations
================================================

There are two ways to define and import novel equations:

* Using pre-existing modular building blocks

* Defining the full dictionary of equations

Using pre-existing modular building blocks
-------------------------------------------

* `teili/models/synapse_models.py and teili/models/neurons_models.py` contain a large number of predefined synapse and neuron models.
* Models are defined using 'NeuronEquationBuilder' and 'SynapseEquationBuilder'. A detailed explanation can be found at :ref:`EquationBuilder.rst`


Defining the full dictionary of equations
--------------------------------------------
.. code-block:: python

  from teili.core.groups import Neurons
  from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder

  builder_object_N = NeuronEquationBuilder.import_eq('path/to/equations/')
  N = Neurons(1, equation_builder= builder_object_N,  name = 'Neurons' )


The file with the equation needs to have the following structure:

* file name: my_equations.py
* dictionary entries: **model**, **threshold**, **reset** and **parameters**
* parameters needs to have the **refP** entrance

.. code-block:: python


  from brian2.units import *
  import numpy as np

  my_equations ={'model':
  '''   ''',
  'threshold':
  '''   ''',
  'reset':
  ''' ''',
  'parameters':
  {	'refP' : '0.*second',
  }}
