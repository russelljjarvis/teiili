from brian2.units import * 
Exponential = {'model':
'''
        dI_syn/dt = (-I_syn) / tausyn + kernel: amp (clock-driven)
        Iin{input_number}_post = I_syn *  sign(weight) : amp (summed)

        kernel : amp * second **-1
        tausyn : second (constant) # synapse time constant
        w_plast : 1
        baseweight : amp (constant)     # synaptic gain
        weight : 1
        
         
         ''',
'on_pre':
'''

        I_syn += baseweight * abs(weight) * w_plast
        
         
         ''',
'on_post':
'''
 
         
         
''',
'parameters':
{
'tausyn' : '5. * msecond',
'w_plast' : '1',
'baseweight' : '1. * namp',
'kernel' : '0. * second ** -1 * amp',
}
}