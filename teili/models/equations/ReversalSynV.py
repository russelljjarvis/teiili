from brian2.units import * 
ReversalSynV = {'model':
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
        
         
         ''',
'on_pre':
'''

        gI += baseweight * abs(weight) * w_plast
        
         
         ''',
'on_post':
'''
 
         
         
''',
'parameters':
{
'gI' : '0. * siemens',
'tausyn' : '5. * msecond',
'EIe' : '60. * mvolt',
'EIi' : '-90. * mvolt',
'w_plast' : '1',
'baseweight' : '7. * nsiemens',
'weight' : '1',
'kernel' : '0. * metre ** -2 * kilogram ** -1 * second ** 2 * amp ** 2',
}
}