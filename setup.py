# -*- coding: utf-8 -*-
# @Author: mmilde
# @Date:   2018-06-19 10:18:33

from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
import pathlib
from pathlib import Path
import os


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def initialize_options(self):
        install.initialize_options(self)

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
        if "readthedocs.org" not in os.getcwd():
            from teili.models import neuron_models, synapse_models

            path = os.path.expanduser("~")

            equation_path = os.path.join(path, "teiliApps", "equations")
            teili_equation_path = os.path.join("teili", "models", "equations")

            neuron_models.main(path=equation_path)
            synapse_models.main(path=equation_path)
            neuron_models.main(path=teili_equation_path)
            synapse_models.main(path=teili_equation_path)

            source_path = os.path.join(os.getcwd(), "tests", "")
            target_path = os.path.join(path, "teiliApps", "unit_tests", "")

            if not os.path.isdir(target_path):
                Path(target_path).mkdir(parents=True)

            os.system('cp {}* {}'.format(source_path, target_path))

            source_path = os.path.join(os.getcwd(), "tutorials", "")
            target_path = os.path.join(path, "teiliApps", "tutorials", "")
            if not os.path.isdir(target_path):
                Path(target_path).mkdir(parents=True)

            os.system('cp {}* {}'.format(source_path, target_path))
        install.run(self)


class PostDevelopCommand(develop):
    """Post-installation for installation mode."""

    def initialize_options(self):
        develop.initialize_options(self)

    def finalize_options(self):
        develop.finalize_options(self)

    def run(self):
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
        if "readthedocs.org" not in os.getcwd():
            from teili.models import neuron_models, synapse_models

            path = os.path.expanduser("~")

            equation_path = os.path.join(path, "teiliApps", "equations")
            teili_equation_path = os.path.join("teili", "models", "equations")

            neuron_models.main(path=equation_path)
            synapse_models.main(path=equation_path)
            neuron_models.main(path=teili_equation_path)
            synapse_models.main(path=teili_equation_path)

            neuron_models.main(path=path)
            synapse_models.main(path=path)

            source_path = os.path.join(os.getcwd(), "tests", "")
            target_path = os.path.join(path, "teiliApps", "unit_tests", "")

            if not os.path.isdir(target_path):
                Path(target_path).mkdir(parents=True)

            os.system('cp {}* {}'.format(source_path, target_path))

            source_path = os.path.join(os.getcwd(), "tutorials", "")
            target_path = os.path.join(path, "teiliApps", "tutorials", "")
            if not os.path.isdir(target_path):
                Path(target_path).mkdir(parents=True)

            os.system('cp {}* {}'.format(source_path, target_path))
        develop.run(self)

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="teili",
    version="1.0.2",
    author="Moritz B. Milde",
    author_email="m.milde@westernsydney.edu.au",
    description=("This toolbox was developed to provide computational  "
                 "neuroscientists and neuromorphic engineers with a "
                 "playground for implementing neural algorithms which "
                 "are simulated using Brian 2."),
    long_description=README,
    long_description_content_type="text/markdown",
    include_package_data=True,
    license="MIT",
    keywords="Neural algorithms, building blocks, Spiking Neural Networks",
    url="https://code.ini.uzh.ch/ncs/teili",
    install_requires=[
        'setuptools>=39.2.0',
        'numpy>=1.15.1',
        'seaborn>=0.8.1',
        'sparse>=0.3.0',
        'Brian2==2.2.2.1',
        'scipy>=1.1.0',
        'pyqtgraph>=0.10.0',
        'pandas>=0.23.1',
        'matplotlib>=2.2.2',
        'h5py>=2.8.0',
        'pyqt5>=5.10.1',
        'sympy==1.5.1'
    ],
    cmdclass={
        'install': PostInstallCommand,
        'develop': PostDevelopCommand,
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
        'teili.tools',
        'teili.tools.visualizer',
        'teili.tools.visualizer.DataControllers',
        'teili.tools.visualizer.DataModels',
        'teili.tools.visualizer.DataViewers',
        'tutorials',
        'tests',
    ],


    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
)
