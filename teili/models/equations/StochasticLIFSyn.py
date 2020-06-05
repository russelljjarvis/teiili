from brian2.units import *
StochasticLIFSyn = {'model': '''
                        dI_syn/dt = int(I_syn*psc_decay + psc_decay_probability)/ms : 1 (clock-driven)
                        Iin{input_number}_post = I_syn * sign(weight)   : 1 (summed)
                        psc_decay = tau_syn/(tau_syn+1.0)                    : 1

                        weight            : 1
                        psc_decay_probability : 1
                        gain_syn : 1
                        tau_syn               : 1 (constant)
                        ''',
                    'on_pre': '''
                        I_syn += gain_syn*weight
                        ''',
                    'on_post': '''

                        ''',
                    'parameters': {
                        'weight' : '1',
                        'gain_syn' : '1',
                        'tau_syn': '10',
                        }
                   }

