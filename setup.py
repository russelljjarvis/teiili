# -*- coding: utf-8 -*-
# @Author: mmilde
# @Date:   2018-06-19 10:18:33

from setuptools import setup

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
        'teili.stimuli',
        'teili.tools'
    ],

    install_requires=[
        'numpy',
        'seaborn',
        'sparse',
        'brian2',
        'scipy',
        'pyqtgraph',
        'pandas',
        'matplotlib',
        'h5py',
        'pyqt5'
    ],

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
