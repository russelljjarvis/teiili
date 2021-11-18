from brian2.units import * 
StochasticSyn = {'model':
'''
        dI_syn/dt = int(I_syn*decay_syn/mA + decay_probability_syn)*mA/second : amp (clock-driven)
        decay_probability_syn = rand() : 1 (constant over dt)
        Iin{input_number}_post = I_syn * sign(weight)                           : amp (summed)

        decay_syn = tausyn/(tausyn + dt) : 1

        weight                : 1
        w_plast               : 1
        gain_syn              : amp
        tausyn               : second (constant)
        
         ''',
'on_pre':
'''

        I_syn += gain_syn * abs(weight) * w_plast
        
         ''',
'on_post':
'''

        
         
''',
'parameters':
{
'weight' : '1',
'w_plast' : '1',
'gain_syn' : '1. * mamp',
'tau_syn' : '3. * msecond',
'lfsr_num_bits_syn' : '6',
}
}
