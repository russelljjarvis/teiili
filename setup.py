# -*- coding: utf-8 -*-
# @Author: mmilde
# @Date:   2018-06-19 10:18:33

from setuptools import setup
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
        'setuptools>=39.2.0',
        'numpy>=1.14.5',
        'seaborn>=0.8.1',
        'sparse>=0.3.0',
        'Brian2>=2.1.3.1',
        'scipy>=1.1.0',
        'pyqtgraph>=0.10.0',
        'pandas>=0.23.1',
        'matplotlib>=2.2.2',
        'h5py>=2.8.0',
        'pyqt5>=5.10.1'
    ],

    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Computational neuroscientists",
        "Intended Audience :: Neuromorphic engineers",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Programming Language :: Python3",
    ],
)

if "readthedocs.org" not in os.getcwd():
    from teili.models import neuron_models, synapse_models

    neuron_models.main()
    synapse_models.main()

    os.system('sudo chown -R $SUDO_USER:$SUDO_USER teili/models/equations/*')
    print("Install successful! Models have been placed into teili/models/equations folder")
