# -*- coding: utf-8 -*-
# @Author: mmilde
# @Date:   2018-06-19 10:18:33

from setuptools import setup
from setuptools.command.install import install
from pathlib import Path
import os


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    user_options = install.user_options + [
        ('dir=', None, 'Specify the path to extract examples, unit_tests and pre-defined equations.'),
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.dir = None

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
        if "readthedocs.org" not in os.getcwd():
            from teili.models import neuron_models, synapse_models
            if self.dir is None:
                print("No path specified, falling back to defaul location: {}". format(
                    os.path.expanduser("~")))
                path = os.path.expanduser("~")
            else:
                path = self.dir

            neuron_models.main(path=path)
            synapse_models.main(path=path)

            source_path = os.path.join(os.getcwd(), "tests", "")
            target_path = os.path.join(path, "teiliApps", "unit_tests", "")

            if not os.path.isdir(target_path):
                Path(target_path).mkdir(parents=True)

            os.system('cp {}* {}'.format(source_path, target_path))

            source_path = os.path.join(os.getcwd(), "examples", "")
            target_path = os.path.join(path, "teiliApps", "examples", "")
            if not os.path.isdir(target_path):
                Path(target_path).mkdir(parents=True)

            os.system('cp {}* {}'.format(source_path, target_path))
        install.run(self)


setup(
    name="teili",
    version="0.2",
    author="Moritz Milde",
    author_email="mmilde@ini.uzh.ch",
    description=("This toolbox was developed to provide computational  "
                 "neuroscientists and neuromorphic engineers with a "
                 "playground for implementing neural algorithms which "
                 "are simulated using Brian 2."),
    license="MIT",
    keywords="Neural algorithms, building blocks, Spiking Neural Networks",
    url="https://code.ini.uzh.ch/ncs/teili",
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
    cmdclass={
        'install': PostInstallCommand
    },
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
