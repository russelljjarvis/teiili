***************
Getting started
***************

Welcome to `teili`!

To start using `teili`, follow the instructions_ below and see our tutorials_.



Prepare a virtual environment
=============================

Before we can install **teili** we encourage to set up a virtual environment using anaconda (or miniconda). Make sure you have installed conda_.

- Create a virtual environment using conda_

.. code-block:: bash

    conda create --name <myenv> python=3.7

.. note:: Replace <myenv> with the desired name for your virtual environment


.. note:: If you want to use CTXCTL_ add the particular python version to the conda environment

.. code-block:: bash

   conda create --name <myenv> python=3.7.1

- Activate your conda environment

.. code-block:: bash

    source activate <myenv>

Install latest stable release of `teili`
========================================
For installing the latest stable release run the following command

.. code-block:: bash

   pip install teili
   python -m teili.tools.generate_teiliApps


The first command will install `teili` and all its dependencies.
The second command generates a ``teiliApps`` directory in your home folder.
This folder contains a selection of neuron and synapse models, tutorials, as well as unit tests.
Please run the unit tests to check if everything is working as expected by

.. code-block:: bash

    cd ~/teiliApps
    python -m unittest discover unit_tests/

.. attention:: Running the ``unit_tests`` will output a lot of ``Warnings`` to your terminal. This, however, does not mean that the ``unit_tests`` failed as we need to generate and kill test plots. As long as the last line states:
   **Ran 78 tests in 93.373s
   OK**
   Everything is good.

**You are good to go**.

.. note:: If you find yourself seeing an warning as shown below consider updating pyqtgraph to the current development version using                    
   **pip install git+https://github.com/pyqtgraph/pyqtgraph@develop**

.. code-block:: bash

    Error in atexit._run_exitfuncs:
    Traceback (most recent call last):
      File "/home/you/miniconda3/envs/teili_test/lib/python3.6/site-packages/pyqtgraph/__init__.py", line 312, in cleanup
        if isinstance(o, QtGui.QGraphicsItem) and isQObjectAlive(o) and o.scene() is None:
    ReferenceError: weakly-referenced object no longer exists


In case you want the latest stable version of `teili` you refer to our repository_
The following steps are only required, if you need the most recent stable version/unstable developments for your simulations. If you do not require the latest version please proceed to tutorials_.


Install latest development version of `teili`
=============================================

- To get the most recent version of `teili` you can either clone the repository_ as shown below or download_ the tar.gz file.

.. code-block:: bash

    git clone https://code.ini.uzh.ch/ncs/teili.git

.. note:: If you have set up git properly you can use of course
   **git clone git@code.ini.uzh.ch:ncs/teili.git**

.. note:: For the **latest development version** of `teili` please checkout the `dev` branch:
   **git checkout dev**.

- Navigate to the parent folder containing the cloned repository or the downloaded ``tar.gz`` file and install teili using pip (make sure you activated your virtual environment).

.. code-block:: bash

    # Point pip to the location of the setup.py
    pip install teili/
    # or point pip to the downloaded tar.gz file
    pip install teili*.tar.gz

.. note:: Note that the *path* provided in the install command needs to point to the folder which contains the **setup.py** file. When using the source files the ``teiliApps`` directory is generated automically.

The ``setup.py`` will by default create a folder in your home directory called ``teiliApps``.
This folder contains a selection of neuron and synapse models, tutorials, as well as unit tests.
Please run the unit tests to check if everything is working as expected by

.. code-block:: bash

    cd ~/teiliApps
    python -m unittest discover unit_tests/


**You are good to go!**

.. note:: Due to `pyqtgraph` the unit tests will print warnings, as we generate and close figures to test the functionality of `teili`. These warning are normal. As longer as no ``Error`` is returned, everything is behaving as expected.
 
If you want to change the location of ``teiliApps``, you can do so by moving the folder manually.

The installation instructions above will install all requirements and dependencies.
It will also build pre-defined neuron and synapse models and place them in ``teiliApps/equations/``.
Make sure you checkout our tutorials_.

Re-building models after installation
=====================================

.. note:: By default models are generated during installation. **Only if** you accidentally deleted them manually you need to rebuild models.

By default the models will be placed in ``teiliApps/equations/``. If you want to place them at a different location follow the instructions below:

.. code-block:: bash

    source activate <myenv>
    python

.. code-block:: python

    from teili import neuron_models, synapse_models
    neuron_models.main("/path/to/my/equations/")
    synapse_models.main("/path/to/my/equations/")

Note, that the following folder structure is generated in the specified location: ``/path/to/my/equations/teiliApps/equations/``.
If you simply call the classes without a path the equations will be placed in ``~/teiliApps/equations/``.
Have a look at our tutorials_ to see how to use teili and which features it provides to you.

.. _conda: https://conda.io/docs/user-guide/install/index.html
.. _tutorials: https://teili.readthedocs.io/en/latest/scripts/Tutorials.html
.. _instructions: https://teili.readthedocs.io/en/latest/scripts/Getting%20started.html#installation
.. _CTXCTL: http://ai-ctx.gitlab.io/ctxctl/index.html
.. _repository: https://code.ini.uzh.ch/ncs/teili
.. _download: https://code.ini.uzh.ch/ncs/teili/repository/archive.tar.gz?ref=dev
