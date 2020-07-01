'''This module defines the model of the stochastic leaky I&F neuron.

The arguments of the function int() should contain units and should be greater
than 1 (otherwise membrane would be always clipped to zero). This is why some
multiplication and divisions with units were added to the equations. The
problem with this approach is that manual intervention is necessary for other
units, e.g. uV.
'''
from brian2.units import * 
StochasticLIF = {'model': '''
                     dVm/dt = (int(not refrac)*int(normal_decay) + int(refrac)*int(refractory_decay))*volt/second : volt
                     normal_decay = (decay_rate*Vm + (1-decay_rate)*(Vrest + Rm*I))/mV + decay_probability : 1
                     refractory_decay = (decay_rate_refrac*Vm + (1-decay_rate_refrac)*Vrest)/mV + decay_probability : 1

                     I = Iexp + Iin + Iconst + Inoise - Iadapt : amp
                     decay_rate = tau/(tau + dt)                      : 1
                     tau = Rm*Cm : second
                     decay_rate_refrac = refrac_tau/(refrac_tau + dt) : 1
                     refrac = Vm<Vrest                                    : boolean

                     decay_probability : 1
                     Rm                : ohm    (constant) # membrane resistance
                     Cm      : farad     (constant)        # membrane capacitance
                     Iconst  : amp                         # constant input current
                     Iexp    : amp                            # exponential current
                     Iadapt  : amp                            # adaptation current
                     Inoise  : amp                            # noise current
                     Iin = Iin0        : amp
                     Iin0 : amp
                     refrac_tau        : second (constant)
                     refP              : second
                     Vthres            : volt   (constant)
                     Vrest             : volt   (constant)
                     Vreset            : volt   (constant)

                     x : 1 (constant) # x position on a 2d grid
                     y : 1 (constant) # y position on a 2d grid
                     lfsr_num_bits : 1 # Number of bits in the LFSR used

                     ''',
                 'threshold': '''Vm>=Vthres''',
                 'reset': '''Vm=Vreset''',
                 'parameters':{
                     'Vthres': '16*mV',
                     'Vrest': '3*mV',
                     'Vreset': '0*mV',
                     'Iexp': '0*pA',
                     'Iadapt': '0*pA',
                     'Inoise': '0*pA',
                     'Iconst': '0*pA',
                     'Cm': '500*uF',
                     'Rm' : '20*ohm',
                     'refrac_tau': '10*ms',
                     'refP': '12.*ms',
                     'lfsr_num_bits': '20'
                     }
                 }
