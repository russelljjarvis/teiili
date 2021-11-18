from brian2.units import * 
StochasticLIFSummed = {'model':
'''
        dVm/dt = (int(not refrac)*int(normal_decay) + int(refrac)*int(refractory_decay))*mV/second : volt
        normal_decay = clip((decay_rate*Vm + (1-decay_rate)*(Vrest + g_psc*I))/mV + decay_probability, Vm_min, Vm_max) : 1
        refractory_decay = (decay_rate_refrac*Vm + (1-decay_rate_refrac)*Vrest)/mV + decay_probability : 1
        decay_probability = rand() : 1 (constant over dt)

        I = clip(Iin + Iconst, -15*mA, 15*mA) : amp
        decay_rate = tau/(tau + dt)                      : 1
        decay_rate_refrac = refrac_tau/(refrac_tau + dt) : 1
        refrac = Vm<Vrest                                    : boolean

        g_psc                : ohm    (constant) # Gain of post synaptic current
        Iconst  : amp                         # constant input current
        tau               : second (constant)
        refrac_tau        : second (constant)
        refP              : second
        Vthr              : volt   (constant)
        Vm_min            : 1      (constant)
        Vm_max            : 1      (constant)
        Vrest             : volt   (constant)
        Vreset            : volt   (constant)
        Vthres : volt

    
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
'Vm_min': '0',
'Vm_max': '16',
}
}
