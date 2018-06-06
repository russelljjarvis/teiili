Welcome to teili's documentation!
=================================
teili, das /taɪli/
swiss german diminutive for piece.

This toolbox was developed to provide computational neuroscientists, as well as neuromorphic engineers, a play ground for neuronally implemented algorithms which are simulated using brian2. By providing pre-defined neural algorithms and an intuitive way to combine different aspects of those algorithms, e.g. plasticity, connectivity etc, we try to shorten the production time of novel neural algorithms. Furthermore, by providing an easy and modular way to construct those algorithms from the basic building blocks of computaton, e.g. neurons and synapses, we aim to reduce the gap between software simulation and hardware emulation

Contributors
^^^^^^^^^^^^
Moritz Milde - Initial work, equation builder, building blocks, neuron and synapse models, testbench -

Alpha Renner - Initial work, core, visualizer, building blocks, tools -
Marco Rasetto - Equation builder -
Renate Krause - Visualization -
Karla Burelo - Synaptic kernels -
Nicoletta Risi - Mismatch, DYNAPSE interface -
Daniele Conti - Silicon Neuron and Synapse -

Getting started
===============

To use the tool
    .. code:: bash

        glit clone https://code.ini.uzh.ch/ncs/teili.git
        cd teili/
        git checkout dev

Install dependencies
    .. code:: bash

        sudo apt install python3 python3-pip python3-matplotlib python3-setuptools cython
        pip3 install brian2 sparse seaborn h5py numpy scipy pyqtgraph pyqt5 easydict

To build default neuron and synapse models
    .. code:: bash

        python -m models/neuron_models.py
        python -m models/synapse_models.py

Checkout more examples on how to use teili in the `examples` folder
    .. include:: neuron_synapse_test.py
        :language: python
        :number-lines:


.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
