# Equation Builder
The equation builder serves as a dynamic model generator. It takes model templates, located in `teili/models/builder/templates/` and combines these template snippets using `teili/models/builder/combine.py`.

There are two distinct equation builder classes:
*  NeuronEquationBuilder
*  SynapseEquationBuilder

Each builder is wrapped by a neuron/synapse model generator class located in `teili/models/`:
*  neuron_model
*  synapse_model

## Keyword arguments for builder
In order to generate a neuron/synapse model, its builder needs to be initialized using specific keywords which define the model itself and thus which template equation/parameters are combined.
### NeuronEuqationBuilder keywords
```python
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
num_inputs = 2
my_neu_model = NeuronEquationBuilder.__init__(base_unit='current', adaptation='calcium_feedback',
                               integration_mode='exponential', leak='leaky',
                               position='spatial', noise='none')
my_neuron.add_input_currents(num_inputs)
```
The keywords explained:
*  base_unit: Indicates whether neuron is current or conductance based.
*  adaptation: What type of adaptive feedback should be used. So far only calciumFeedback is implemented.
*  integration_mode: Determines whether integration up to spike-generation is linear or exponential.
*  leak: Enables leaky integration.
*  position: To enable spatial-like position indices on neuron.
*  noise: **NOT YET IMPLMENTED!** This will in the future allow independent mismatch-like noise to be added to each neuron.
*  refractory: Refractory period of the neuron.

### SynapseEquationBuilder keywords
```python
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
my_syn_model = SynapseEquationBuilder.__init__(base_unit='DPI',
                                               plasticity='non_plastic')
```
The keywords explained:
*  base_unit: Indicates whether synapse is current-based, conductance-based or a DPI current model.
*  kernel: Specifying temporal kernel with which each spike gets convolved, i.e. exponential decay, or alpha function.
*  plasticity: Plasticity algorithm for the synaptic weight. Can either be 'non_plastic', 'fusi' or 'stdp'.


## Dictionary structure
Both equation builders have a dictionary attribute which keys represent the respective necessary keywords to generate a neuron/synapse model, in order to simulate it using brian2.

The keywords, given to the EquationBuilder class are used to select template dictionaries which are combined.
This is done by passing these keywords to `current_equation_sets` and `current_parameters` in case of neurons and to `modes`, `kernels`, `plasticity_models` and `current_parameters`.
```python
# In case of neurons
keywords = combine_neu_dict(eq_templ, param_templ)
# In case of synapses
keywords = combine_syn_dict(eq_tmpl, param_templ)
```

### Neuron model keywords
The dictionary `keywords` has the following keys:
```python
keywords = {'model': keywords['model'],
            'threshold': keywords['threshold'],
            'reset': keywords['reset'],
            'refractory': 'refP',
            'parameters': keywords['parameters']}

```
### Synapse model keywords
The dictionary `keywords` has the following keys:
```python
keywords = {'model': keywords['model'],
            'on_pre': keywords['on_pre'],
            'on_post': keywords['on_post'],
            'parameters': keywords['parameters']}
```

## Class methods
### import_eq
A function to import pre-defined neuron_model. This function can load a dictionary and its keywords in order to initialize the EquationBuilder.
```python
from teili.models.builder.neuron_equation_builder import NeuronEquationBuilder
my_neu_model = NeuronEquationBuilder.import_eq(
    'teili/models/equations/DPI', num_inputs=2)
```
where num_inputs specifies how many distinct neuron population project to the target population.

For synapses the import works as follows:
```python
from teili.models.builder.synapse_equation_builder import SynapseEquationBuilder
my_syn_model = SynapseEquationBuilder.import_eq(
    'teili/models/equations/DPISyn')
```
### export_eq
In order to generate models, which can later be changed manually and imported again the EuqationBuilder class features an export method which can be used as:
```python
path = os.path.dirname(os.path.realpath(teili.models.__file__))
DPI = NeuronEquationBuilder.__init__(base_unit='current', adaptation='calcium_feedback',
                                     integration_mode='exponential', leak='leaky',
                                     position='spatial', noise='none')
DPI.add_input_currents(num_inputs)
DPI.export_eq(os.path.join(path, "DPI"))
```
For synapse models:
```
path = os.path.dirname(os.path.realpath(teili.models.__file__))
dpiSyn = SynapseEquationBuilder.__init__(base_unit='DPI',
                                         plasticity='non_plastic')

dpiSyn.export_eq(os.path.join(path, "DPISyn"))
```

## var_replacer
This function takes two equation sets in form of strings and replaces all lines which start with '%'.
```python
'%x = theta' --> 'x = theta'
'%x' --> ''
```
This feature allows equations that we don't want to compute to be removed from the template by writing '%[variable]' in the other equation blocks.

To replace variables and lines:
```python
from teili.models.builder.combine import var_replacer
var_replacer(first_eq, second_eq, params)
```

