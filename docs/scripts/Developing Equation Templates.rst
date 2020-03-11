Developing Equation Templates
=============================

For using existing models please refer to our `neuron and synapse tutorials`_
If you need a different model or you want to test a new idea it easy to use all of `teili's` functionality with your custom model.
There are two ways to test, develop and use your custom model

1. Defining a static static dictionary
2. Defining a set of templates

The first way allows you to quickly define a model inside a `.py` file as a dictionary.
This way you can easily debug and test your new idea with fiddling with the `EquationBuilder` class.
You can use the `import_eq` method to import your custm model into `teili`.
See below_ for more details of how to create your custom dictionary.
The second way is to define a template of your model and use the `EquationBuilder` class to dynamically build your model.
This is a bit more tricky, but has the advantage that others can potentially use your model in the future.
The second advantage is that you might just want to add a custom learning rule or a set of additional equations to calculate some proxy values, required for learning.
So you intend to use the existing model, but want to change it's e.g. learning dynamics.
'Teili` provides a `combine_equation` method exactly for this case.
For more details of how to do so, please see here_


Defining static models (import_eq)
-------------------------------------------

To create a custom model (**locally**) please follow these steps:

1. Create a file in your desired directory: ``my/awesome/path/<my_cool_model.py>``
2. Import all necessary units from `brian2`
3. Make a dictionary with the same name as your file: ``my_cool_model = {}``
4. Create four keys in case of a neuron model (`model`, `threshold`, `reset` and `parameters`) or in case of a synapse model (`model`, `on_pre`, `on_post` and `parameters`).

.. attention:: The name of the file and the name of the dictionary need to be same (without the `.py` extension).

The file (``my/awesome/path/<my_cool_model.py>``) should now look like this for **neuron model**:

.. code-block:: python

  from brian2.units import *

  my_cool_model = {
      'model':'''
          ''',
      'threshold':'''
          ''',
      'reset':'''
          ''',
      'parameters':{
          'refP' : '0.*second',
          }
      }
  
.. attention:: The neuron parameters dictionary **needs** at least a entry for the refractory period ``refP``.

The file (``my/awesome/path/<my_cool_model.py>``) should look like this for a **synapse model**

.. code-block:: python

  from brian2.units import *

  my_cool_model = {
      'model':'''
          ''',
      'on_pre':'''
          ''',
      'on_post':'''
          ''',
      'parameters':{
          }
      }

Once you filled your dictionary with your model and/or standard model + custom equations, you can use the ``import_eq`` method to start using it.

.. code-block:: python

  from teili.core.groups import Neurons
  from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder

  my_cool_neuron_model = NeuronEquationBuilder.import_eq('my/awesome/path/<my_cool_model.py>')
  N = Neurons(1, equation_builder=my_cool_neuron_model,  name='my_cool_neuron' )

.. note:: Please make sure you remove '<' and '>' from your strings.


Create new templates for dynamic model generation
--------------------------------------------------

As described above the second way to create and use your custom model is to extend the provided neuron/synapse templates and the ``neuron_models`` or ``synapse_models`` respectively.

.. attention:: To add new templates you have to fork and clone the repository. Details of how to contribute are given below. We highly recommend building and test your custom model using the static model import method described above, before fiddling with the dynamic model generation.


Neuronal templates
******************

Navigate to the template sub-directory (``teili/models/builder/templates``) and open ``neuron_templates.py``.
As described above the neuron model is defined as a dictionary in which the following ``keys`` are required:
* 'model'
* 'threshold'
* 'reset'

The parameters are defined **separately** here.
your new entry to ``neuron_templates`` should look like this:

.. code-block:: python

  #As an example lets define a new voltage-based model
  new_neuron_model = {
    'model': '''
        ''',
    'threshold': ''' ''',
    'reset':'''
        '''
    }

  #define new parameters
  new_neuron_model_params = {
      }

At the end of the file, we need to associate the newly defined model and its parameters with ``keywords`` used by the ``EquiationBuilder``.
Each neuron model as two corresponding dictionaries.
The first one is the ``equation_sets`` dictionary which depending on the ``base_unit`` is either called ``current_equation_sets`` or ``voltage_equation_sets``.
The second dictionary is the ``parameters`` dictionary which depending on the ``base_unit`` is either called ``current_parameters`` or ``voltage_parameters``.
The ``key`` in this dictionary needs to match die ``**kwargs`` given in the class initilisation and the value needs to match the name of the dictionary defined in ``neuron_templates.py``. 
That allows the ``EquationBuilder`` upon initialisation to dynamically assemble the respective equations into a coherent model.
This functionality is especially useful when you e.g. just createdevelop a new threshold adaption mechanism, a different adaption current dynamics or a new plasticity rule.
After the template was added the newly defined neuron model must be added to ``neuron_models.py`` to be generated dynamically.
The entry should look similar to this

.. code-block:: python

   class my_neuron_model(NeuronEquationBuilder):
       """This class provides you with all equations to simulate a current-based
       awesome model...
       """
 
       def __init__(self, num_inputs=1):
           NeuronEquationBuilder.__init__(self, base_unit='current', adaptation='none',
                                         integration_mode='exponential', leak='leaky',
                                         position='spatial', awesome_new_feature='adp')
           self.add_input_currents(num_inputs)

.. note:: This is just an example.

Synapes templates
*****************

Navigate to the template sub-directory (``teili/models/builder/templates``) and open ``synapse_templates.py``.
* Define your new model equations and the corresponding parameters in synapse models.
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
 
Once the templates are extendend you can add the model to ``synapse_models.py`` located in ``teili/models/``.

.. code-block:: python

  # Define the new model
  class my_model(SynapseEquationBuilder):
      def __init__(self):
          SynapseEquationBuilder.__init__(self, base_unit='current', kernel = 'new_kernel')

   my_model = my_model()
   my_model.export_eq(os.path.join(path, "my_model"))



Create a new template using the unit_less dictionary
****************************************************

You might want to develop, test or define a plasticity mechanism or part of a neuron model which neither uses currents or voltages.
Therefore, we provide a third dictionary called ``unit_less``.

* ``unit_less`` models follow the same proceduce as ``current`` or ``voltage`` based dictionaries.
* **Parameters** to be defined to the ``unit_less`` dictionary while the model needs to be added to one of the model dictionaries.
* When creating the synapse model **base_unit** should be defined as **unit_less**.
* This can be usefull to define learning rules that involve gain modulation or activity modulation. (e.g. ``STDGM``)

.. code-block:: python

  # Define the new model
  class my_model(SynapseEquationBuilder):
      def __init__(self):
          SynapseEquationBuilder.__init__(self, base_unit='unit_less')


Combine equations and replace variables
***************************************

A major strength of `teili` is its **modularity**.
This starts already at the equation level.
If you want to to e.g. test an existing synapse model with let's say Spike-Timing Dependent Plasticity (STDP), you don't need to specify a complete new model.
Instead you can initialise the ``EquationBuilder``  differently, such that the STDP template is combined with a ``DPI`` synapse using a ``Alpha`` -shaped kernel.
Internally, the different equation sets are combined depending on the provided keyword arguments and equations which have a default definition but are defined differently in a given plasticity mechanism are replaced.
We provide a method call ``var_replacer`` which uses the '%' symbol to replace equations in the original set of equations.
Compare e.g. the ``synapse_models``: **Alpha** and **AlphaStdp** (located in ``teili/models/synapapse_models.py``) and their respective templates (located in ``teili/models/builder/templates/synapse_templates.py``)
For more information see our documentation on the `equation builder`_


How to contribute and publish your custom model
------------------------------------------------

Once you tested your custom model locally, you can add a new template.
To do so you need to work directly inside the library code.
Head to the repository_ and do the following steps

1. Fork the ``dev`` branch
2. Clone the forked repository to your local system
3. Create a new branch within your forked repository (e.g. `git checkout -b new_branch` + `git remote add upstream URL_of_project`)
4. Follow the steps from our `contribution guide`_
5. Make your changes, add appropriate unite tests and push it to your repository.

You can create a pull request to add your remote change to `teili`

* Click `compare & pull request` button on github.
* Click `create pull request` to open a new pull request
* Wait for us to approve it and give you feedback :)

.. _neuron and synapse tutorials: https://teili.readthedocs.io/en/latest/scripts/Tutorials.html#dynamic-model-generation-vs-static-model-import
.. _here: 
.. _contribution guide: https://teili.readthedocs.io/en/latest/scripts/Developing%20Equation%20templates.html
.. _equation builder: https://teili.readthedocs.io/en/latest/scripts/Equation%20builder.html#var-replacer
