from brian2.units import * 
StochasticLIF = {'model': '''
dVmem/dt = int(not refrac)*int(decay_rate*Vmem + (1-decay_rate)*Vin + decay_probability)/ms + int(refrac)*int(decay_rate_refrac*Vmem + (1-decay_rate_refrac)*Vrest + decay_probability)/ms :  1
#dVmem/dt = int(decay_rate*Vmem + (1-decay_rate)*Vin + decay_probability)/ms :  1 (unless refractory)
                              decay_rate = tau/(tau+1.0) : 1
                              decay_rate_refrac = refrac_tau/(refrac_tau+1.0) : 1
                              refrac = Vmem<Vrest : 1

                              decay_probability : 1
                              tau : 1 (constant)
                              refrac_tau : 1 (constant)
                              Vthres : 1 (constant)
                              Vrest : 1 (constant)
                              Vreset : 1 (constant)
                              Vin : 1 (constant)
                              refP : second
                          ''',
                 'threshold': '''Vmem>=Vthres''',
                 'reset': '''Vmem=Vreset''',
                 'parameters': {'Vthres': '16',
                                'Vrest': '3',
                                'Vreset': '0',
                                'tau': '19',
                                'refrac_tau': '10',
                                'Vin': '20',
                                'refP': '12.*msecond'
                                }
                 }
