from brian2.units import * 
StochLIFAdapt = {'model':
'''
        dVm/dt = (int(not refrac)*int(normal_decay) + int(refrac)*int(refractory_decay))*mV/second : volt
        normal_decay = clip((decay_rate*Vm + (1-decay_rate)*(Vrest + g_psc*I))/mV + decay_probability, 3, 16) : 1
        refractory_decay = (decay_rate_refrac*Vm + (1-decay_rate_refrac)*Vrest)/mV + decay_probability : 1
        decay_probability = lfsr_timedarray( ((seed+t) % lfsr_max_value) + lfsr_init ) / (2**lfsr_num_bits): 1
        dVthres/dt = Vthres*decay_thresh/second : volt

        I = Iin + Iconst : amp
        decay_rate = tau/(tau + dt)                      : 1
        decay_rate_refrac = refrac_tau/(refrac_tau + dt) : 1
        decay_thresh = tau_thres/(tau_thres + dt) : 1
        refrac = Vm<Vrest                                    : boolean

        lfsr_max_value : second
        seed : second
        lfsr_init : second
        g_psc                : ohm    (constant) # Gain of post synaptic current
        Iconst  : amp                         # constant input current
        tau               : second (constant)
        tau_thres         : second (constant)
        refrac_tau        : second (constant)
        refP              : second
        Vrest             : volt   (constant)
        Vreset            : volt   (constant)
        update_counter    : second
        update_time       : second (constant)
        theta             : volt (constant)

        lfsr_num_bits : 1 # Number of bits in the LFSR used

    
        x : 1 (constant) # x location on 2d grid
        y : 1 (constant) # y location on 2d grid
        


         Iin = Iin0  : amp # input currents

         Iin0 : amp
''',
'threshold':
'''Vm>=Vthres''',
'reset':
'''Vm=Vreset
   Vthres = clip(Vthres+0.1*mV, 0, 16*mV)''',
'parameters':
{
'Vthres' : '10. * mvolt',
'Vrest' : '3. * mvolt',
'Vreset' : '0. * volt',
'Iconst' : '0. * amp',
'g_psc' : '1. * ohm',
'tau' : '19. * msecond',
'tau_thres' : '60000. * msecond',
'refrac_tau' : '2. * msecond',
'refP' : '0. * second',
'lfsr_num_bits' : '6',
'update_time' : '1000 * msecond',
'theta' : '8 * mvolt',
'update_counter' : '0 * msecond',
}
}
