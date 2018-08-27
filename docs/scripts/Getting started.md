# Getting started

Welcome to teili!

To start using teili, follow the instructions below.
```
git clone https://code.ini.uzh.ch/ncs/teili.git
```

## Install python requirements
```
sudo apt install python3 python3-pip
```
and now install teili
```
sudo pip3 install teili/
```
This will install all requirements and dependencies.
It will also build pre-defined neuron and synapse models and
place them in `teili/models/equations`.
Note that the path provided in the install command needs to point to the folder, which contains the `setup.py` file.
To uninstall teili just execute the following command
```
sudo pip3 uninstall teili/
```

In case you want to re-build the pre-defined models you need to navigate to the `model` folder:
```
cd teili/models/
```
and run the following two scripts:
```
python3 -m models/neuron_models.py
python3 -m models/synapse_models.py
```

Have a look at our `examples` to see different use cases and tutorials.

