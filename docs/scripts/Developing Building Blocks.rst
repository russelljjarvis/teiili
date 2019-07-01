
***************
Developing Building Blocks
***************
This section explains the generation of new and more complex ``BuildingBlocks``:
we provide users with an example of this in ``octa-HierarchicalBB.py``

Every building block inherits from the class BuildingBlocks which has attributes
such as ``sub_blocks``, ``input_groups``, ``output_groups`` and ``hidden_groups``.

Recommended practices for creating custom building blocks go as follow:

- Keep the class initialization as concise as possible
- Create a generation function which implements the desired connectivity
- Label correctly ``sub_blocks``, ``input_groups``, ``output_groups`` and ``hidden_groups``
- Remember to _set_tags as you expand the network

Important Notes:

- When running the network, add the newly generated BB as well as all the sub_blocks the network depends on

.. code-block:: python
  Net.add(
          test_OCTA,
          test_OCTA.sub_blocks['predictionWTA'],
          test_OCTA.sub_blocks['compressionWTA']
        )

- There is a fundamental difference between the ``groups`` and ``_groups`` attribute. ``groups``
will return all groups present in the overall building block while ``_groups`` will return
only the groups specific to that building block.

- When overwriting an existing population group in one of the sub_blocks, remember to initialize
 all the connections and the monitors regarding that group. Which will now be specific to
 the parent class.
