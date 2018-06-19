# -*- coding: utf-8 -*-
# @Author: mmilde
# @Date:   2018-06-19 10:18:33

from setuptools import setup
from teili.models import neuron_models, synapse_models
import os

setup(
    name="teili",
    version="0.2",
    author="Moritz Milde",
    author_email="mmilde@ini.uzh.ch",
    description=("This toolbox was developed to provide computational  "
                 "neuroscientists and neuromorphic engineers with a "
                 "playground for implementing neural algorithms which "
                 "are simulated using brian2."),
    license="MIT",
    keywords="Neural algorithms, building blocks, Spiking Neurla Networks",
    url="https://code.ini.uzh.ch/ncs/teili",
    packages=[
        'teili',
        'teili.core',
        'teili.models',
        'teili.models.builder',
        'teili.models.builder.templates',
        'teili.models.equations',
        'teili.models.parameters',
        'teili.building_blocks',
        'teili.stimuli',
        'teili.tools'
    ],

    install_requires=[
        'numpy>=1.13.0',
        'seaborn>=0.8.1',
        'sparse>=0.3.0',
        'brian2>=2.1.3.1',
        'scipy>=1.0.1',
        'pyqtgraph>=0.10.0',
        'pandas',
        'matplotlib>=1.5.1',
        'h5py',
        'pyqt5'
    ],

    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
    ],
)

neuron_models.main()
synapse_models.main()

os.system('sudo chown -R $SUDO_USER:$SUDO_USER teili/models/equations/*')
