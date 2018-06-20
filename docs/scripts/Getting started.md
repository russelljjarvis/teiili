# Getting started

Welcome to teili!

To start using teili follow the instructions below.
```
git clone https://code.ini.uzh.ch/ncs/teili.git
cd teili/
```

## Install python requirements
```
sudo apt install python3 python3-pip
```
and now install teili
```
sudo python3 setup.py install
```
This will install all dependencies, as well as requirements.
Furthermore, it will build pre-defined neuron and synapse models and
place them in `teili/models/equations`.

In case you want to re-build the pre-defined models you need to navigate to the `model` folder
```
cd teili/models/
```
and run the following two scripts:
```
python3 -m models/neuron_models.py
python3 -m models/synapse_models.py
```

Have a look into our `examples` to see different use cases and tutorials.
