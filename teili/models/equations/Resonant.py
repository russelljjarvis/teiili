from brian2.units import * 
Resonant = {'model':
'''
        dI_syn/dt = (-I_syn) / tausyn + kernel: amp (clock-driven)
        Iin{input_number}_post = I_syn *  sign(weight) : amp (summed)

        kernel  = s * omega : amp * second **-1
        tausyn : second (constant) # synapse time constant
        w_plast : 1
        baseweight : amp (constant)     # synaptic gain
        weight : 1
        

        ds/dt = -s/tausyn - I_syn*omega : amp (clock-driven)

        omega: 1/second
        tausyn_kernel : second
        
         ''',
'on_pre':
'''

        I_syn += 0 * amp
        
        s += baseweight * w_plast * abs(weight)

        
         ''',
'on_post':
'''
  
         
''',
'parameters':
{
'tausyn' : '0.5 * msecond',
'w_plast' : '1',
'baseweight' : '1. * namp',
'omega' : '3. * khertz',
'tausyn_kernel' : '0.5 * msecond',
}
}