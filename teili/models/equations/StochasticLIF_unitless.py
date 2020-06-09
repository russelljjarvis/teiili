from brian2.units import * 
StochasticLIFul = {'model': '''
                     # Charging cell
                     #dVm/dt = int(not refrac)*int(decay_rate*Vm + (1-decay_rate)*Vin + decay_probability)/ms + int(refrac)*int(decay_rate_refrac*Vm + (1-decay_rate_refrac)*Vrest + decay_probability)/ms :  1
                     # Refractory period alternative
                     #dVm/dt = int(decay_rate*Vm + (1-decay_rate)*Vin + decay_probability)/ms :  1 (unless refractory)
                     # Input current
                     dVm/dt = int(not refrac)*int(normal_decay)/ms + int(refrac)*int(refractory_decay)/ms :  1
                     normal_decay = decay_rate*Vm + (1-decay_rate)*Vin + decay_probability + input_gain*Iin/amp : 1
                     refractory_decay = decay_rate_refrac*Vm + (1-decay_rate_refrac)*Vrest + decay_probability : 1

                     decay_rate = tau/(tau+1.0) : 1
                     decay_rate_refrac = refrac_tau/(refrac_tau+1.0) : 1
                     refrac = Vm<Vrest : boolean

                     decay_probability : 1
                     input_gain : 1
                     Iin = Iin0  : 1 # input currents
                     Iin0 : amp
                     tau : 1 (constant)
                     refrac_tau : 1 (constant)
                     Vthres : 1 (constant)
                     Vrest : 1 (constant)
                     Vreset : 1 (constant)
                     Vin : 1 (constant)
                     refP : second
                     ''',
                 'threshold': '''Vm>=Vthres''',
                 'reset': '''Vm=Vreset''',
                 'parameters':{
                     'Vthres': '16',
                     'Vrest': '3',
                     'Vreset': '0',
                     'tau': '19',
                     'input_gain' : '1',
                     'refrac_tau': '10',
                     'Vin': '3',
                     'refP': '12.*msecond'
                     }
                 }

