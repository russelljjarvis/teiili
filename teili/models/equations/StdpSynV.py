from brian2.units import * 
StdpSynV = {'model':
'''
        dgI/dt = (-gI) / tausyn + kernel     : siemens (clock-driven)
        I_syn = gI*(EI - Vm_post)            : amp
        Iin{input_number}_post = I_syn *  sign(weight)  : amp (summed)

        EI =  EIe                            : volt        # reversal potential
        kernel                               : siemens * second **-1
        tausyn                               : second   (constant) # synapse time constant
        w_plast                              : 1
        baseweight                           : siemens (constant)     # synaptic gain
        weight                               : 1
        EIe                                  : volt
        EIi                                  : volt
        
         
        dApre/dt = -Apre / taupre : 1 (event-driven)
        dApost/dt = -Apost / taupost : 1 (event-driven)
        w_max: 1 (constant)
        taupre : second (constant)
        taupost : second (constant)
        dApre : 1 (constant)
        Q_diffAPrePost : 1 (constant)
        ''',
'on_pre':
'''

        gI += baseweight * abs(weight) * w_plast
        
         
        Apre += dApre*w_max
        w_plast = clip(w_plast + Apost, 0, w_max)
        ''',
'on_post':
'''
 
         
        Apost += -dApre * (taupre / taupost) * Q_diffAPrePost * w_max
        w_plast = clip(w_plast + Apre, 0, w_max)
        
''',
'parameters':
{
'gI' : '0. * siemens',
'tausyn' : '5. * msecond',
'EIe' : '60. * mvolt',
'EIi' : '-90. * mvolt',
'w_plast' : '0',
'baseweight' : '7. * nsiemens',
'weight' : '1',
'kernel' : '0. * metre ** -2 * kilogram ** -1 * second ** 2 * amp ** 2',
'taupre' : '20. * msecond',
'taupost' : '20. * msecond',
'w_max' : '0.01',
'diffApre' : '0.01',
'Q_diffAPrePost' : '1.05',
}
}