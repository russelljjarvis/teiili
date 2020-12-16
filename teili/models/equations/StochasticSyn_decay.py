from brian2.units import * 
StochasticSyn_decay = {'model':
'''
        dI_syn/dt = int(I_syn*decay_syn/mA + decay_probability_syn)*mA/second : amp (clock-driven)
        decay_probability_syn = lfsr_timedarray( ((seed_syn+t) % lfsr_max_value_syn) + lfsr_init_syn ) / (2**lfsr_num_bits_syn): 1
        Iin{input_number}_post = I_syn * sign(weight)                           : amp (summed)

        decay_syn = tau_syn/(tau_syn + dt) : 1

        weight                : 1
        w_plast               : 1
        lfsr_max_value_syn : second
        seed_syn : second
        lfsr_init_syn : second
        gain_syn              : amp
        tau_syn               : second (constant)
        lfsr_num_bits_syn : 1 # Number of bits in the LFSR used
        
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