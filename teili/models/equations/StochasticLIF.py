from brian2.units import * 
StochasticLIF = {'model': '''
                     # Charging cell
                     #dVm/dt = int(not refrac)*int(decay_rate*Vm + (1-decay_rate)*Vin + decay_probability)/ms + int(refrac)*int(decay_rate_refrac*Vm + (1-decay_rate_refrac)*Vrest + decay_probability)/ms :  1
                     # Refractory period alternative
                     #dVm/dt = int(decay_rate*Vm + (1-decay_rate)*Vin + decay_probability)/ms :  1 (unless refractory)
                     # Input current
                     dVm/dt = (int(not refrac)*int(normal_decay) + int(refrac)*int(refractory_decay))*volt/second : volt
                     normal_decay = (decay_rate*Vm + (1-decay_rate)*Vin + input_gain*Iin)/mV + decay_probability : 1
                     refractory_decay = (decay_rate_refrac*Vm + (1-decay_rate_refrac)*Vrest)/mV + decay_probability : 1

                     decay_rate = tau/(tau + 1.0*ms)                      : 1
                     decay_rate_refrac = refrac_tau/(refrac_tau + 1.0*ms) : 1
                     refrac = Vm<Vrest                                    : boolean

                     decay_probability : 1
                     input_gain        : ohm
                     Iin = Iin0        : amp
                     Iin0 : amp
                     tau               : second (constant)
                     refrac_tau        : second (constant)
                     refP              : second
                     Vthres            : volt   (constant)
                     Vrest             : volt   (constant)
                     Vreset            : volt   (constant)
                     Vin               : volt   (constant)

                     x : 1 (constant) # x position on a 2d grid
                     y : 1 (constant) # y position on a 2d grid

                     ''',
                 'threshold': '''Vm>=Vthres''',
                 'reset': '''Vm=Vreset''',
                 'parameters':{
                     'Vthres': '16*mV',
                     'Vrest': '3*mV',
                     'Vreset': '0*mV',
                     'tau': '19*ms',
                     'input_gain' : '1*kohm',
                     'refrac_tau': '10*ms',
                     'Vin': '3*mV',
                     'refP': '12.*ms'
                     }
                 }
