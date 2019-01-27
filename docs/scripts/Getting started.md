# Getting started

Welcome to teili!

To start using teili, follow the instructions below.
```
git clone https://code.ini.uzh.ch/ncs/teili.git
```

## Install python requirements
### Installation

*  Checkout the [repository](https://code.ini.uzh.ch/ncs/teili) or [download](https://code.ini.uzh.ch/ncs/teili) the tar.gz file<br />
    ```
    git clone git@code.ini.uzh.ch:ncs/teili.git
    ```
<div class="alert alert-info">

**Note:** Curently the master branch is broken, please use dev by doing `git checkout dev`.

</div>
* Create a virtual environment using [conda](https://conda.io/docs/user-guide/install/index.html)
    ```
    # Replace myenv with the desired name for your virtual environment
    conda create --name myenv python=3.5
    ```
  If you want to use a specific version, as needed e.g. to use [CTXLCTL](http://ai-ctx.gitlab.io/ctxctl/index.html) add the particular python version to the conda environment
   ```
   conda create --name myenv python=3.7.1
   ```

*  Activate your conda environment
    ```
    source activate myenv
    ```

*  Navigate to the parent folder containing the cloned repository or the downloaded `tar.gz` file
    ```
    cd Downloads/
    ```
*  Install teili using pip
    ```
    # Point pip to the location of the setup.py
    pip install teili/
    # or point pip to the downloaded tar.gz file
    pip install teili*.tar.gz
    ```
    The `setup.py` will by default create a folder in your home directory called `teiliApps`.
    This folder contains a selection of neuron and synapse models, example scripts, as well as unit tests.
    Please run the unit tests to check if everything is working as expected by:
    ```
    cd ~/teiliApps
    python -m unittest discover unit_tests/
    ```

    You are good to go!<br />
    If you want to change the location of `teiliApps`, you can do so by moving the folder manually.

This will install all requirements and dependencies.
It will also build pre-defined neuron and synapse models and
place them in `~/teiliApps/equations/`.
Note that the path provided in the install command needs to point to the folder, which contains the `setup.py` file.


### Alternative installation **NOT RECOMMENDED**
    If, however, you want to install all dependencies separately you can run the following commands **NOT RECOMMENDED**:
    ```
    git clone git@code.ini.uzh.ch:ncs/teili.git
    sudo apt install python3 python3-pip, python3-matplotlib python3-setuptools cython
    pip3 install brian2 sparse seaborn h5py numpy scipy pyqtgraph pyqt5 easydict
    ```
    if you did **not** use the setup.py you need to update your `$PYTHONPATH`:

    You can add the following line to your `~/.bashrc`<sup>1</sup>:
    ```
    export PYTHONPATH=$PYTHONPATH:"/path/to/parent_folder/of/teili"
    ```

### Re-building models after installation
<div class="alert alert-info">

**Note:** By default models are generated during installation. Only if you accidentally deleted them manually you need to rebuild models.

</div>


In case you want to re-build the pre-defined models you need to navigate to the `model` folder:
```
cd teili/models/
source activate myenv
```
and run the following two scripts (if you want to use the default location `/home/you/`):
```
python -m neuron_models
python -m synapse_models
```
By default the models will be placed in `~/teiliApps/equations`. If you want to place them at a different location follow the instructions below:
```
source activate myenv
python
from teili import neuron_models, synapse_models
neuron_models.main("/path/to/my/equations/")
synapse_models.main("/path/to/my/equations/")
```
Note, that the following folder structure is generated in the specified location: `/path/to/my/equations/teiliApps/equations/`.
Have a look at our [tutorials](https://teili.readthedocs.io/en/latest/scripts/Tutorials.html) to see how to use teili and which features it provides to you.

