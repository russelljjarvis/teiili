# Neuron and Synapse models
This folder will contain all supported neuron and synapse models, such as normal DPI synapse, DPI with STDP functionality or conductance neuron models.
Files stored in this folder are excluded from git.
To generate files run:
```
cd models/
python3 neuron_models.py
python3 synapse_models.py
```

The created .py files contain dictionaries, describing the model, the reset and threshold behaviour in case of neuron, and the on_pre and on_post behaviour in case of synapses.

You can manually modify these generated .py files according to your needs. If you want to rest all of them run the code snippet above again and everything will be default.

To use the local modified dictionaries you can dow the following in case of neurons:
```
TBA from examples
```

and in case of synapses:
```
TBA from examples
```