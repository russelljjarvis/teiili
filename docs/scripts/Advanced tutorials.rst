Advanced tutorials
==================

WTA live plot
-------------

DVS visualizer
--------------

Sequence learning
-----------------

Teili2Genn
----------

Using the already existing brian2genn_ we can generate ``GeNN`` code which can be executed on a nVidia graphics card.
Make sure to change the ``DPIsyn`` model located in ``teiliApps/equations/DPIsyn.py``. To be able to use brian2genn_ with ``TeiliNetwork``
change this line:

.. code-block:: python
   
   Iin{input_number}_post = I_syn * sign(weight)  : amp (summed)

to

.. code-block:: python

   Iin{input_number}_post = I_syn * (-1 * (weight<0) + 1 * (weight>0))  : amp (summed)

After you made the change in ``teiliApps/equations/DPIsyn.py`` you can run the ``teili2genn_tutorial.py`` located in ``teiliApps/tutorials/``.
The ``TeiliNetwork`` is the same as in ``neuron_synapse_tutorial`` but with the specific commands to use the **genn-backend**.

.. _brian2genn: https://github.com/brian-team/brian2genn
