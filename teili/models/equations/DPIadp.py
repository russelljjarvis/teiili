from brian2.units import * 
DPIadp = {'model':
'''
        dI_syn/dt = (-I_syn - I_gain + 2*Io_syn*(I_syn<=Io_syn))/(tausyn*((I_gain/I_syn)+1)) : amp (clock-driven)
        Iin{input_number}_post = I_syn *  sign(weight)           : amp (summed)
        Iw = abs(weight) * baseweight                            : amp
        I_gain = Io_syn*(I_syn<=Io_syn) + I_th*(I_syn>Io_syn)    : amp
        Itau_syn = Io_syn*(I_syn<=Io_syn) + I_tau*(I_syn>Io_syn) : amp
        tausyn = Csyn * Ut_syn /(kappa_syn * Itau_syn)           : second
        kappa_syn = (kn_syn + kp_syn) / 2                        : 1

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
        
        inh_learning_rate: 1 (constant, shared)
        variance_th: 1 (constant)
        delta_w : 1
        ''',
'on_pre':
'''

        I_syn += Iw * w_plast * I_gain / (Itau_syn * ((I_gain/I_syn)+1))
        
        delta_w = inh_learning_rate * (normalized_activity_proxy_post - variance_th)
        w_plast = clip(w_plast + delta_w, 0, 1.0)
        ''',
'on_post':
'''
  
''',
'parameters':
{
'Io_syn' : '0.5 * pamp',
'kn_syn' : '0.75',
'kp_syn' : '0.66',
'Ut_syn' : '25. * mvolt',
'Csyn' : '1.5 * pfarad',
'I_tau' : '10. * pamp',
'I_th' : '10. * pamp',
'I_syn' : '0.5 * pamp',
'w_plast' : '1',
'baseweight' : '7. * pamp',
'weight' : '1',
'inh_learning_rate' : '0.01',
'variance_th' : '0.67',
}
}