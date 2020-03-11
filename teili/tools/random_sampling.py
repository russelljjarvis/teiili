#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module provides functions to sample from a random distribution

All functions in here should be callable in a similar way as rand() or randn() that are part of brian2

Please note: brian2 uses a specific random generator in the 'randomkit', so any seed you set in brian2 will not necessarily
apply here, depending on which rng is used. (The truncated randn uses brian2's rand, so it's fine, gamma however uses another one)
If you need a seed, it may be easy to implement though. (TODO!)

Please also note that, currently, there is a bug in brian2 (issue #988) that does not allow you to use the several functions
with the same dependencies for the same variable (but this probably happens only rarely).

I used brian2/input/binomial.py as a template
"""
# @author: alpha

import numpy as np
import os
from scipy.stats import truncnorm
from brian2 import DEFAULT_FUNCTIONS

from brian2 import check_units

import matplotlib.pyplot as plt
from brian2 import NeuronGroup, prefs, set_device, run, ms, mV, seed

from brian2 import Nameable, Function
from brian2.utils.stringtools import replace


def _randn_trunc_generate_cpp_code(lower, upper, name):
    # C++ implementation
    cpp_code = '''
    float %NAME%(const int _vectorisation_idx) {
        float retVal = 0; 
        do {retVal = _randn(_vectorisation_idx);
        } while ((retVal > %UPPER%) || (retVal < %LOWER%));
        return retVal;
    }
    '''
    cpp_code = replace(cpp_code, {'%NAME%': name, '%UPPER%': str(upper), '%LOWER%': str(lower)})
    cpp_code = cpp_code.replace('inf','std::numeric_limits<float>::max()')
    dependencies = {'_randn': DEFAULT_FUNCTIONS['randn']}
    return {'support_code': cpp_code}, dependencies


class Randn_trunc(Function, Nameable):
    """
    Sample from a truncated Gaussian
    We are using this in core/groups to add mismatch.
    In python it wraps truncnorm.rvs(lower, upper, size=N)

    refer to the example below
    """
    implementations = {
        'cpp': _randn_trunc_generate_cpp_code,
    }

    @check_units(lower=1, upper=1)
    def __init__(self, lower, upper, name='_randn_trunc*'):
        Nameable.__init__(self, name)

        def sample_function(vectorisation_idx):
            try:
                N = len(vectorisation_idx)
            except TypeError:
                N = int(vectorisation_idx)
            return truncnorm.rvs(lower, upper, size=N)

        try:
            Function.__init__(self, pyfunc=lambda: sample_function(1),
                              arg_units=[], return_unit=1, stateless=False,
                              auto_vectorise=True)
        except TypeError as e:
            # this is necessary for backward compatibility with brian2 < 2.3, as the argument auto_vectorise
            # does not exist
            Function.__init__(self, pyfunc=lambda: sample_function(1),
                              arg_units=[], return_unit=1, stateless=False)

        self.implementations.add_implementation('numpy', sample_function)

        for target, func in Randn_trunc.implementations.items():
            code, dependencies = func(lower=lower, upper=upper, name=self.name)
            # print('target:', target, '\ncode: ', code, '\ndependencies: ', dependencies, '\nname:', self.name)
            self.implementations.add_implementation(target, code,
                                                    dependencies=dependencies,
                                                    name=self.name)


def _rand_gamma_generate_cpp_code(alpha, beta, name):
    # C++ implementation
    cpp_code = '''
    std::mt19937 rng(std::random_device{}());
    // Not ideal, but probably good enough for us:
    // https://codereview.stackexchange.com/questions/109260/seed-stdmt19937-from-stdrandom-device  
    // Would be good to seed the rng with a random number from brian2, so the brian2 seed affects the rng here.
    float %NAME%(const int _vectorisation_idx) {
        std::gamma_distribution<double> distribution(%ALPHA%,1/%BETA%);
        float retVal = distribution(rng);
    	return retVal;
    }
    '''
    cpp_code = replace(cpp_code, {'%NAME%': name, '%BETA%': str(beta), '%ALPHA%': str(alpha)})
    dependencies = {}
    return {'support_code': cpp_code}, dependencies


class Rand_gamma(Function, Nameable):
    """
    Sample from a gamma distribution.
    Refer to the example below.
    """
    prefs.codegen.cpp.headers += ['<random>']

    implementations = {
        'cpp': _rand_gamma_generate_cpp_code,
    }

    @check_units(alpha=1, beta=1)
    def __init__(self, alpha, beta, name='_rand_gamma*'):
        Nameable.__init__(self, name)

        def sample_function(vectorisation_idx):
            try:
                N = len(vectorisation_idx)
            except TypeError:
                N = int(vectorisation_idx)
            f = -1 if beta < 0 else 1
            if N == 1:
                return f * np.random.gamma(alpha, scale=f / beta)
            else:
                return f * np.random.gamma(alpha, scale=f / beta, size=N)

        try:
            Function.__init__(self, pyfunc=lambda: sample_function(1),
                              arg_units=[], return_unit=1, stateless=False,
                              auto_vectorise=True)
        except TypeError:
            # this is necessary for backward compatibility with brian2 < 2.3, as the argument auto_vectorise
            # does not exist
            Function.__init__(self, pyfunc=lambda: sample_function(1),
                              arg_units=[], return_unit=1, stateless=False)

        self.implementations.add_implementation('numpy', sample_function)

        for target, func in Rand_gamma.implementations.items():
            code, dependencies = func(alpha=alpha, beta=beta, name=self.name)
            self.implementations.add_implementation(target, code,
                                                    dependencies=dependencies,
                                                    name=self.name)


if __name__ == '__main__':
    # Some examples how to use the gamma sampling
    # And some different parametrizations

    n_samples = 10000

    # outside of brian2
    rand_gamma = Rand_gamma(2, 2)

    gamma_samples = [rand_gamma() for _ in range(n_samples)]

    plt.figure()
    _ = plt.hist(gamma_samples, 50)
    plt.show()

    # brian2 with numpy codegen
    prefs.codegen.target = "numpy"

    ng = NeuronGroup(n_samples, 'testvar : 1')
    ng.testvar = 'rand_gamma()'

    plt.figure()
    plt.hist(ng.testvar, 50)
    plt.show()

    # keep std constant
    gamma_samples = [[Rand_gamma(alpha, np.sqrt(alpha))() for _ in range(n_samples)] for alpha in range(1, 20, 2)]
    plt.figure()
    _ = plt.hist(gamma_samples, 500, histtype='step')
    plt.show()

    print(np.mean(gamma_samples, 1))
    print(np.std(gamma_samples, 1))

    # set mean and std like in normal dist
    std = 0.2 * mV
    mu = -0.4 * mV  #
    alpha = (1 / std ** 2) * mu ** 2
    beta = (1 / std ** 2) * mu * 1000 * mV
    gamma_samples = [Rand_gamma(alpha, beta)() * 1000 for _ in range(n_samples)]
    plt.figure()
    _ = plt.hist(gamma_samples, 500, histtype='step', density=True)
    plt.show()

    print(np.mean(gamma_samples))
    print(np.std(gamma_samples))
    print(np.var(gamma_samples))

    # %% It also works in standalone mode:
    standaloneDir = os.path.expanduser('~/gamma_standalone')
    set_device('cpp_standalone', directory=standaloneDir, build_on_run=True)
    seed(42)  # does not affect sampling from gamma distribution!


    ng = NeuronGroup(n_samples, '''
    testvar : 1
    testvar2 : 1''', name = 'ng_test')
    ng.namespace.update({
        'rand_gamma': Rand_gamma(4.60, -10750.0),
        'randn_trunc': Randn_trunc(-1.5,1.5)
    })
    ng.testvar = 'rand_gamma()'
    ng.testvar2 = '5*randn_trunc()'

    run(10 * ms)

    plt.figure()
    plt.title('rand_gamma')
    plt.hist(ng.testvar, 50, histtype='step')
    plt.show()

    plt.figure()
    plt.title('randn_trunc')
    plt.hist(ng.testvar2, 50, histtype='step')
    plt.show()
