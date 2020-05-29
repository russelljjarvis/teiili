from brian2.units import * 
ReversalSynVfusi = {'model':
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
        
         
        dCa/dt = (-Ca/tau_ca)                    : volt (event-driven)

        updrift = 1.0*(w>theta_w)                : 1
        downdrift = 1.0*(w<=theta_w)             : 1
        dw/dt = (alpha*updrift)-(beta*downdrift) : 1 (event-driven)

        wplus       : 1
        wminus      : 1
        theta_upl   : volt     (constant)
        theta_uph   : volt     (constant)
        theta_downh : volt     (constant)
        theta_downl : volt     (constant)
        theta_V     : volt     (constant)
        alpha       : 1/second (constant)
        beta        : 1/second (constant)
        tau_ca      : second   (constant)
        w_min       : 1        (constant)
        w_max       : 1        (constant)
        theta_w     : 1        (constant)
        w_ca        : volt     (constant)
        ''',
'on_pre':
'''

        gI += baseweight * abs(weight) * w_plast
        
         
        up = 1. * (Vm_post>theta_V) * (Ca>theta_upl) * (Ca<theta_uph)
        down = 1. * (Vm_post<theta_V) * (Ca>theta_downl) * (Ca<theta_downh)
        w += wplus * up - wminus * down
        w = clip(w,w_min,w_max)
        w_plast = floor(w+0.5)
        ''',
'on_post':
'''
 
         
        Ca += w_ca
        
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
'wplus' : '0.2',
'wminus' : '0.2',
'theta_upl' : '180. * mvolt',
'theta_uph' : '1. * volt',
'theta_downh' : '90. * mvolt',
'theta_downl' : '50. * mvolt',
'theta_V' : '-59. * mvolt',
'alpha' : '100. * uhertz',
'beta' : '100. * uhertz',
'tau_ca' : '8. * msecond',
'w_ca' : '250. * mvolt',
'w_min' : '0',
'w_max' : '1',
'theta_w' : '0.5',
'w' : '0',
}
}