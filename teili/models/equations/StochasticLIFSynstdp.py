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
                        dApre/dt = -Apre / taupre : 1 (event-driven)
                        dApost/dt = -Apost / taupost : 1 (event-driven)

                        weight                : 1
                        w_plast               : 1
                        w_max: 1 (constant)
                        taupre : second (constant)
                        taupost : second (constant)
                        Q_diffAPrePost : 1 (constant)
                        psc_decay_probability : 1
                        gain_syn              : amp
                        tau_syn               : second (constant)
                        ''',
                    'on_pre': '''
                        I_syn += gain_syn*weight
                        Apre += Apre*w_max
                        w_plast = clip(w_plast + Apost, 0, w_max)
                        ''',
                    'on_post': '''
                        Apost += -Apre * (taupre / taupost) * Q_diffAPrePost
                        w_plast = clip(w_plast + Apre, 0, w_max)

                        ''',
                    'parameters': {
                        'weight' : '1',
                        'w_plast' : '0',
                        'gain_syn' : '1*mA',
                        'tau_syn': '3*ms',
                        'taupre' : '10. * msecond',
                        'taupost' : '10. * msecond',
                        'w_max' : '1.0',
                        'Apre' : '0.1',
                        'Q_diffAPrePost' : '1.05',
                        }
                   }

