# Getting started

Welcome to teili!

To use teili
```
git clone https://code.ini.uzh.ch/ncs/teili.git
cd teili/
git checkout dev
```

## Install dependencies
```
sudo apt install python3 python3-pip python3-matplotlib python3-setuptools cython
pip3 install brian2 sparse seaborn h5py numpy scipy pyqtgraph pyqt5 easydict
```
## Building pre-defined models

To build default neuron and synapse models
```
python -m models/neuron_models.py
python -m models/synapse_models.py
```