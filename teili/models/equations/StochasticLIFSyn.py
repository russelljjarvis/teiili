'''This module defines the model of the stochastic leaky I&F neuron's synapse.

The arguments of the function int() should contain units and should be greater
than 1 (otherwise membrane would be always clipped to zero). This is why some
multiplication and divisions with units were added to the equations. The
problem with this approach is that manual intervention is necessary for other
units, e.g. nA.
'''
from brian2.units import *
StochasticLIFSyn = {'model': '''
                        dI_syn/dt = int(I_syn*psc_decay/mA + psc_decay_probability)*amp/second : amp (clock-driven)
                        Iin{input_number}_post = I_syn * sign(weight)                           : amp (summed)

                        psc_decay = tau_syn/(tau_syn+1.0*ms) : 1

                        weight                : 1
                        w_plast               : 1
                        psc_decay_probability : 1
                        gain_syn              : amp
                        tau_syn               : second (constant)
                        ''',
                    'on_pre': '''
                        I_syn += gain_syn*weight
                        ''',
                    'on_post': '''

                        ''',
                    'parameters': {
                        'weight' : '1',
                        'w_plast' : '0',
                        'gain_syn' : '1*mA',
                        'tau_syn': '10*ms',
                        }
                   }

