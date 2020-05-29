from brian2.units import * 
DPIShunt = {'model':
'''
        dI_syn/dt = (-I_syn - I_gain + 2*Io_syn*(I_syn<=Io_syn))/(tausyn*((I_gain/I_syn)+1)) : amp (clock-driven)
        Ishunt{input_number}_post = I_syn *  sign(weight)  * (weight<0) : amp (summed)
        Iw = abs(weight) * baseweight                                   : amp
        I_gain = Io_syn*(I_syn<=Io_syn) + I_th*(I_syn>Io_syn)           : amp
        Itau_syn = Io_syn*(I_syn<=Io_syn) + I_tau*(I_syn>Io_syn)        : amp
        tausyn = Csyn * Ut_syn /(kappa_syn * Itau_syn)                  : second
        kappa_syn = (kn_syn + kp_syn) / 2                               : 1

        weight     : 1
        w_plast    : 1
        baseweight : amp   (constant)
        I_tau      : amp   (constant)
        I_th       : amp   (constant)
        kn_syn     : 1     (constant)
        kp_syn     : 1     (constant)
        Ut_syn     : volt  (constant)
        Io_syn     : amp   (constant)
        Csyn       : farad (constant)
        
         ''',
'on_pre':
'''

        I_syn += Iw * w_plast * I_gain * (weight<0)/(Itau_syn*((I_gain/I_syn)+1
        
         ''',
'on_post':
'''
 
         
''',
'parameters':
{
'Csyn' : '1.5 * pfarad',
'Io_syn' : '0.5 * pamp',
'I_tau' : '10. * pamp',
'Ut_syn' : '25. * mvolt',
'baseweight' : '50. * pamp',
'weight' : '1',
'kn_syn' : '0.75',
'kp_syn' : '0.66',
'I_th' : '10. * pamp',
'I_syn' : '0.5 * pamp',
}
}