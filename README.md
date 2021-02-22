<p align="center">
 <a href='https://teili.readthedocs.io/en/latest/?badge=latest' alt="Documentation Status">
    <img src='https://readthedocs.org/projects/teili/badge/?version=latest' /></a>
 <a href="https://gitlab.com/neuroinf/teili/commits/dev" alt="build status">
    <img src="https://code.ini.uzh.ch/ncs/teili/badges/dev/build.svg" /></a>
 <a href="https://gitlab.com/neuroinf/teili/-/commits/dev" alt="coverage report">
    <img src="https://gitlab.com/neuroinf/teili/badges/dev/coverage.svg" /></a>
 <a href="https://gitlab.com/neuroinf/teili/-/commits/master" alt="pipeline">
    <img src="https://gitlab.com/neuroinf/teili/badges/master/pipeline.svg" /></a>
</p>

# teili

teili, das /taɪli/, Swiss german diminutive for piece. <br />

This toolbox was developed to provide computational neuroscientists and neuromorphic engineers with a playground for implementing neural algorithms which are simulated using **brian2**.<br />
Please report issues via the gitlab [issue tracker](https://gitlab.com/neuroinf/teili/-/issues). You can find the documentation [here](https://teili.readthedocs.io/en/latest/).


By providing pre-defined neural algorithms, a contributing guide to create novel neural algorithms and an intuitive way to combine different aspects of those neural algorithms, e.g. plasticity, connectivity etc, we try to shorten the development time required to test and implement novel neural algorithms and hierarchically assemble them.
Furthermore, by providing an easy and modular way to construct those algorithms from the basic building blocks of computaton, e.g. neurons, synapses and EI-networks, we aim to promote the development of reproducible computational models and networks thereof. 
Additionally we aim to reduce the gap between software simulation and specific hardware emulation/implementation by an easy way to switch neuron/synapse models in a highly complex building block of a developed neural algorithm.

## Getting Started

This toolbox is still in its alpha phase, so if you want to use it, follow the install instructions below.
Please also contact the main contributors with feedback.

Please look at the examples here: `~/teiliApps/tutorials/` after successfully installing **teili** and our [Documentation](https://teili.readthedocs.io/en/latest/)

### Installation

* Create a virtual environment using [conda](https://conda.io/docs/user-guide/install/index.html)
    ``` bash
    # Replace myenv with the desired name for your virtual environment
    conda create --name myenv python=3.7
    ```
  If you want to use a specific version, as needed e.g. to use [CTXLCTL](http://ai-ctx.gitlab.io/ctxctl/index.html) add the particular python version to the conda environment
   ``` bash
   conda create --name myenv python=3.6.6
   ```

*  Activate your conda environment
    ``` bash
    source activate myenv
    ```

*  If you want to use the stable release simply run:

    ```
    pip install teili
    ```
*  To get all tutorials, unit tests and statically defined neuron and synapse models please run (simply inside your terminal)
    ```
    python -m teili.tools.generate_teiliApps
    ```

*  If you want to work with the latest version of **teili** clone the [repository](https://gitlab.com/neuroinf/teili.git) or [download](https://gitlab.com/neuroinf/teili/-/archive/master/teili-master.tar.gz) the tar.gz file<br />
    ``` bash
    git clone https://gitlab.com/neuroinf/teili.git
    ```
*  Navigate to the parent folder containing the cloned repository or the downloaded `tar.gz` file
    ```
    cd Downloads/
    ```
*  Install teili using pip
    ``` bash
    # Point pip to the location of the setup.py
    pip install teili/
    # or point pip to the downloaded tar.gz file
    pip install teili*.tar.gz
    ```
    The `setup.py` will by default create a folder in your home directory called `teiliApps` (if you use the source files teiliApps will be automocally generated. In case of the stable release from pypi.org you need to manually call the function to generate teiliApps as described above).
    This folder contains a selection of neuron and synapse models, example scripts, as well as unit tests.
    Please run the unit tests to check if everything is working. As we test also plotting functionality of teili we generate and kill plotting windows which causes warnings.

    ``` bash
    cd ~/teiliApps/
    python -m unittest discover unit_tests/
    ```

If you run the above command and the last line states ``Ran 78 tests in 93.373 OK``, everything is good. You are good to go!<br />

We would recommend using [iPython](https://pypi.org/project/ipython/) with [Spyder](https://www.spyder-ide.org/) or[Jupyter](https://pypi.org/project/jupyter/) as IDE, but any other editior/IDE is fine as well.

You probably need to use Linux if you want to use standalone code generation,
otherwise, Windows and Mac OSX works fine.

## Basic usage

``` python
from brian2 import ms
from teili import Neurons, Connections
from teili.models.neuron_models import DPI
from teili.models.synapse_models import DPISyn

# how to create a Neuron
num_neurons = 10
refP = 3 * ms
Neuron1 = Neurons(num_neurons, equation_builder=DPI(numInputs=1),
                  refractory=refP, name='Neuron1')

# how to create a Synapse
Synapse1 = Connections(Neuron1, Neuron1,
                      equation_builder=DPISyn(),
                      method='euler',
                      name='Synapse1')
```
For a more detailed explanation have a look at our [Tutorial](https://teili.readthedocs.io/en/latest/scripts/Tutorials.html)

### Tutorials
Please look at the [Neuron & Synapse tutorial](https://teili.readthedocs.io/en/latest/scripts/Tutorials.html#neuron-synapse-tutorial), which is located in `~/teiliApps/tutorial/`.
You can also use them to test your installation.
To run an example and test if eveything is working, run the following command

``` bash
cd ~/teiliApps/examples/
python3 neuron_synapse_test.py
```
For more examples and use cases have look at our [Documentation](https://teili.readthedocs.io/en/latest/index.html)


## Brian2 debugging tips
Simulation is not going as expected?
* Restart Python kernel
* Are all groups added to the network?
* Are all statevars initialized with the correct value? (e.g. Membrane potential with resting potential, not 0)
* Use group.print() in order to see the equations
* Use connections.plot() in order to get a visualization



## Authors
See here for a [list of the authors](https://teili.readthedocs.io/en/latest/scripts/Contributors.html).


## License
_teili_ is licenced under the MIT license, see the `LICENSE` file.

