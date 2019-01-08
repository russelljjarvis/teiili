# -*- coding: utf-8 -*-
# @Author: mmilde
# @Date:   2018-06-19 10:18:33

from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
from pathlib import Path
import os
import atexit


def _post_install():
    print('Preparing models...')
    if "readthedocs.org" not in os.getcwd():
        from teili.models import neuron_models, synapse_models
        path = os.path.expanduser("~")
        equation_path = os.path.join(path, "teiliApps", "equations")

        neuron_models.main(path=equation_path)
        synapse_models.main(path=equation_path)

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


class new_install(install):
    def __init__(self, *args, **kwargs):
        super(new_install, self).__init__(*args, **kwargs)
        atexit.register(_post_install)


class new_develop(develop):
    def __init__(self, *args, **kwargs):
        super(new_develop, self).__init__(*args, **kwargs)
        atexit.register(_post_install)


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
        'install': new_install,
        'develop': new_develop,
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
