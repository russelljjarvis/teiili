from brian2.units import *
StochasticLIFSyn = {'model': '''
                        dI_syn/dt = int(I_syn*psc_decay/amp + psc_decay_probability)*amp/second : amp (clock-driven)
                        Iin{input_number}_post = I_syn * sign(weight)                           : amp (summed)

                        psc_decay = tau_syn/(tau_syn+1.0*ms) : 1

                        weight                : 1
                        w_plast               : 1
                        psc_decay_probability : 1
                        gain_syn              : amp
                        tau_syn               : second (constant)
                        ''',
                    'on_pre': '''
                        I_syn += gain_syn*weight
                        ''',
                    'on_post': '''

                        ''',
                    'parameters': {
                        'weight' : '1',
                        'w_plast' : '0',
                        'gain_syn' : '1*mA',
                        'tau_syn': '10*ms',
                        }
                   }

