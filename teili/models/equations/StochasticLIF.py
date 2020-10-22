from brian2.units import * 
StochasticLIF = {'model':
'''
        dVm/dt = (int(not refrac)*int(normal_decay) + int(refrac)*int(refractory_decay))*mV/second : volt
        normal_decay = (decay_rate*Vm + (1-decay_rate)*(Vrest + g_psc*I))/mV + decay_probability : 1
        refractory_decay = (decay_rate_refrac*Vm + (1-decay_rate_refrac)*Vrest)/mV + decay_probability : 1

        I = Iin + Iconst : amp
        decay_rate = tau/(tau + dt)                      : 1
        decay_rate_refrac = refrac_tau/(refrac_tau + dt) : 1
        refrac = Vm<Vrest                                    : boolean

        decay_probability : 1
        g_psc                : ohm    (constant) # Gain of post synaptic current
        Iconst  : amp                         # constant input current
        tau               : second (constant)
        refrac_tau        : second (constant)
        refP              : second
        Vthres            : volt   (constant)
        Vrest             : volt   (constant)
        Vreset            : volt   (constant)

        lfsr_num_bits : 1 # Number of bits in the LFSR used

    
        x : 1 (constant) # x location on 2d grid
        y : 1 (constant) # y location on 2d grid
        


         Iin = Iin0  : amp # input currents

         Iin0 : amp
''',
'threshold':
'''Vm>=Vthres''',
'reset':
'''Vm=Vreset ''',
'parameters':
{
'Vthres' : '16. * mvolt',
'Vrest' : '3. * mvolt',
'Vreset' : '0. * volt',
'Iconst' : '0. * amp',
'g_psc' : '1. * ohm',
'tau' : '19. * msecond',
'refrac_tau' : '2. * msecond',
'refP' : '0. * second',
'lfsr_num_bits' : '6',
}
}