# Building Blocks

TBA description of Building blocks

## Winner-takes-all (WTA)
Every building block has a set of parameters such as weights and refractory period, which can be specified outside the building block gneration and unpacked to the building block. For the WTA building_block this dictionary looks as follows:
```
wtaParams = {'weInpWTA': 100,
             'weWTAInh': 55,
             'wiInhWTA': -300,
             'weWTAWTA': 45,
             'sigm': 2,
             'rpWTA': 1 * ms,
             'rpInh': 1 * ms,
             'wiInhInh': -20,
             'EI_connection_probability': 0.5,
             'IE_connection_probability': 0.66,
             'II_connection_probability': 0.1
```
where each key is defined as:
*  weInpWTA: Excitatory synaptic weight between input SpikeGenerator and WTA neurons.
*  weWTAInh: Excitatory synaptic weight between WTA population and inhibitory interneuron.
*  wiInhWTA: Inhibitory synaptic weight between inhibitory interneuron and WTA population.
*  weWTAWTA: Self-excitatory synaptic weight (WTA).
*  sigm: Standard deviation in number of neurons for Gaussian connectivity kernel.
*  rpWTA: Refractory period of WTA neurons.
*  rpInh: Refractory period of inhibitory neurons.
*  wiInhInh: Self-inhibitory weight of the interneuron population.
*  EI_connection_probability: WTA to interneuron connectivity probability.
*  IE_connection_probability: Interneuron to WTA connectivity probability
*  II_connection_probability: Interneuron to Interneuron connectivity probability.

Now we can import the necessary modules and build our building block.
```
from teili.building_blocks.wta import WTA
from teili.models.neuron_models import DPI
```
### 1D
The WTA building block comes in two slightly differen versions. The versions only differ in the dimensionality of the WTA.
```
# The number of neurons in your WTA population.
# Note that this number is squared in in the 2D WTA
num_neurons = 50
# The number of neurons which project to your WTA.
# Note that this number is squared in in the 2D WTA
num_input_neurons = 50
my_wta = WTA(name='my_wta', dimensions=1,
             neuron_eq_builder=DPI,
             num_neurons=num_neurons, num_inh_neurons=int(num_neurons**2/4),
             num_input_neurons=num_input_neurons, num_inputs=2,
             block_params=wtaParams,
             monitor=True)
```
### 2D
To generate a 2 dimensional WTA population you can do the following.
```
# The number of neurons in your WTA population.
# Note that this number is squared in in the 2D WTA
num_neurons = 7
# The number of neurons which project to your WTA.
# Note that this number is squared in in the 2D WTA
num_input_neurons = 10
my_wta = WTA(name='my_wta', dimensions=2,
             neuron_eq_builder=DPI,
             num_neurons=num_neurons, num_inh_neurons=int(num_neurons**2/4),
             num_input_neurons=num_input_neurons, num_inputs=2,
             block_params=wtaParams,
             monitor=True)
```


Changing a certain synapse group from being static to plastic:
```
from teili.core.groups import Connections
from teili.models.synapse_models import DPIstdp
my_wta.Groups['synWTAWTA1e'] = Connections(my_wta.Groups['gWTAGroup'],
                                           my_wta.Groups['gWTAGroup'],
                                           equation_builder=DPIstdp
                                           method='euler',
                                           name=my_wta.Groups['synWTAWTA1e'].name)
my_wta.Groups['synWTAWTA1e'].connect(True)
```
Now we changed the standard DPI synapse for the recurrent connection within a WTA population to an All-to-All STDP-based DPI synapse. In order to initialize the plastic weight `w_plast` we need to do:
```
my_wta.Groups['synWTAWTA1e'].weight = 45
my_wta.Groups['synWTAWTA1e'].namespace.update({'w_mean': 0.45})
my_wta.Groups['synWTAWTA1e'].namespace.update({'w_std': 0.35})
# Initializing the plastic weight randomly
my_wta.Groups['synWTAWTA1e'].w_plast = 'w_mean + randn() * w_std'
```

## Chain

## Sequence learning

## Threeway network