Step-by-step guide to creating new equations and contribute
============================================================

There are different ways to define and import novel equations:

* Using pre-existing modular building blocks
* Creating a template using new neuron/synapse models
* Defining the full dictionary of equations

Using pre-existing modular building blocks
-------------------------------------------

* `teili/models/synapse_models.py and teili/models/neurons_models.py` contain a large number of predefined synapse and neuron models.
* Models are defined using 'NeuronEquationBuilder' and 'SynapseEquationBuilder'. A detailed explanation can be found at :ref:`EquationBuilder.rst`


Create a new template using new neuron/synapse models
-------------------------------------------------------

* Define your new model equations and the corresponding parameters in neuron and synapse models.
* Neuron models have the following keywords: **model**, **threshold**, **reset** and **parameters**
* Synapse models have the following keywords: **model**, **on_pre**, **on_post** and **parameters**
* Make sure that both the new model equations and the corresponding parameters are added in the `Dictionary of keywords` at the bottom of the file.
* Neuron templates are divided into two main modes: **current** and **voltage** based equations. Each mode supports equations and parameters.
* Synapse templates are divided based on function. Equations are divided based on **modes**, **kernels**, **plasticity_models** and new **synaptic equations**. 
* Synaptic **modes** are further devided into subcategories: **current**, **DPI**, **conductance**, **DPIShunting** or **unit_less**.
* Create your model using the Neuron or SynapseEquationBuilder.

.. code-block:: python

  #As an example lets define a new current based kernel and create a new model
  new_synapse_kernel = {
    'model': '''

        ''',
    'on_pre': '''

        ''',
    'on_post':
        '''
        '''
    }

  #define new parameters
  new_synapse_kernel_params = {
      }

  #include model in equations
  kernels = {
        'new': new_synapse_kernel
        }

  #include parameters
  current_parameters = {
    'new_kernel': new_synapse_kernel_params,
    }

  # Define the new model
  class my_model(SynapseEquationBuilder):
      def __init__(self):
          SynapseEquationBuilder.__init__(self, base_unit='current', kernel = 'new_kernel')

   my_model = my_model()
   my_model.export_eq(os.path.join(path, "my_model"))



Create a new template using the unit_less dictionary
--------------------------------------------------------

* Same proceduce as in :ref:`Create a new template using new neuron/synapse models`
* Parameters to be defined to the unit_less dictionary while the model needs to be added to one of the model dictionaries.
* When creating the synapse model **base_unit** should be defined as **unit_less**.
* This can be usefull to define learning rules that involve gain modulation or activity modulation. (e.g. `STDGM`)

.. code-block:: python

  # Define the new model
  class my_model(SynapseEquationBuilder):
      def __init__(self):
          SynapseEquationBuilder.__init__(self, base_unit='unit_less')


Combine equations and replace variables
-------------------------------------------------------
* Using the **var_replacer** function you can substitute specific strings in a parameter dictionary.
* For docstrings consult :ref:`Equation Builder`

.. code-block:: python

  from teili.models.builder.combine import var_replacer
  var_replacer(first_eq, second_eq, params)



  



Defining the full dictionary of equations
--------------------------------------------

The file with the neuron equations needs to have the following structure:

* the file name needs to be the same as the dictionary name. In the example below it would be: my_neuron_equations.py
* dictionary entries are: **model**, **threshold**, **reset** and **parameters**
* **parameters** needs to have the **refP** entrance

The file with the synaptic equations needs to have the following structure:

* the file name needs to be the same as the dictionary name.
* dictionary entries are: **model**, **on_pre**, **on_post** and **parameters**

.. code-block:: python

  from teili.core.groups import Neurons
  from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder

  builder_object_N = NeuronEquationBuilder.import_eq('path/to/equations/my_equations.py')
  N = Neurons(1, equation_builder= builder_object_N,  name = 'Neurons' )


The `path/to/equations/my_equations.py` file is as follows:

.. code-block:: python

  from brian2.units import *
  import numpy as np

  my_neuron_equations ={'model':
  '''   ''',
  'threshold':
  '''   ''',
  'reset':
  ''' ''',
  'parameters':
  {	'refP' : '0.*second',
  }}


I want to create my own models and contribute
-------------------------------------------------------------------------

* Fork it.
* Clone it to your local system.
* Make a new branch (e.g. `git checkout -b new_branch` + `git remote add upstream URL_of_project`)
* Make your changes and push it to your repository. Details on how to add neuronal models are explained above.
* Click `compare & pull request` button on github.
* Click `create pull request` to open a new pull request
* Wait for us to approve it and give you feedback :)
