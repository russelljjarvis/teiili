# NCSBrian2Lib

This library was made to facilitate your Brian2 simulations

## Getting Started

This Library is still in alpha phase, so if you want to use it, just clone it and
make sure to add it to your working directory or path
Please also contact the main contributors with feedback.

Please look at the Examples here: https://code.ini.uzh.ch/ncs/NCSBrian2Examples

### Prerequisites

You need to have brian2 installed.

If you use Anaconda, just use

```
conda install brian2
```

We would recommend to use iPython with spyder or jupyter

You probably need to use Linux if you want to use standalone code generation,
otherwise, Windows works fine

### Installing

```
Placeholder
```

### Usage

```
from NCSBrian2Lib import Neurons, Connections

# how to create a Neuron
Neuron1 = Neurons(numNeurons, NeuronEquation, NeuronParams,
                  refractory=refP, name='Neuron1', numInputs=1)

# how to create a Synapse
...
```

## Examples

Please look at the Examples here: https://code.ini.uzh.ch/ncs/NCSBrian2Examples
You can also use them to test your installation


## Brian2 debugging tips
Simulation is not going as expected
* Restart Python kernel
* Are all groups added to the network?
* Are all statevars initialized with the correct value? (e.g. Membrane potential with resting potential, not 0)
* Use group.print() in order to see the equations
* Use connections.plot() in order to get a visualization



## Authors

* **Moritz Milde** - *Initial work* -
* **Alpha Renner** - *Initial work* -
* **Daniele Conti** - *Silicon Neuron and Synapse* -


## License



## Acknowledgments

