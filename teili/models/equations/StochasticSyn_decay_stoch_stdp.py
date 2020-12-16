from brian2.units import * 
StochasticSyn_decay_stoch_stdp = {'model':
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
        
        dApre/dt = int(Apre * decay_stdp_Apre + decay_probability_Apre)/second : 1 (clock-driven)
        dApost/dt = int(Apost * decay_stdp_Apost + decay_probability_Apost)/second : 1 (clock-driven)
        decay_probability_Apre = lfsr_timedarray( ((seed_Apre+t) % lfsr_max_value_Apre) + lfsr_init_Apre ) / (2**lfsr_num_bits_Apre): 1
        decay_probability_Apost = lfsr_timedarray( ((seed_Apost+t) % lfsr_max_value_Apost) + lfsr_init_Apost ) / (2**lfsr_num_bits_Apost): 1

        decay_stdp_Apre = taupre/(taupre + dt) : 1
        decay_stdp_Apost = taupost/(taupost + dt) : 1

        seed_Apre : second
        lfsr_max_value_Apre : second
        lfsr_init_Apre : second
        lfsr_num_bits_Apre : 1
        seed_Apost : second
        lfsr_max_value_Apost : second
        lfsr_init_Apost : second
        lfsr_num_bits_Apost : 1
        w_max: 1 (constant)
        A_max: 1 (constant)
        dApre: 1 (constant)
        A_gain: 1 (constant)
        taupre : second (constant)
        taupost : second (constant)
        ''',
'on_pre':
'''

        I_syn += gain_syn * abs(weight) * w_plast
        
        Apre += dApre
        Apre = clip(Apre, 0, A_max)
        w_plast = int(clip(w_plast - Apost/A_gain*int(lastspike_post!=lastspike_pre), 0, w_max))
        ''',
'on_post':
'''

        
        Apost += dApre
        Apost = clip(Apost, 0, A_max)
        w_plast = int(clip(w_plast + Apre/A_gain*int(lastspike_post!=lastspike_pre), 0, w_max))
        
''',
'parameters':
{
'weight' : '1',
'w_plast' : '1',
'gain_syn' : '1. * mamp',
'tau_syn' : '3. * msecond',
'lfsr_num_bits_syn' : '6',
'taupre' : '3. * msecond',
'taupost' : '3. * msecond',
'w_max' : '15',
'A_max' : '15',
'A_gain' : '4',
'dApre' : '15',
'lfsr_num_bits_Apre' : '6',
'lfsr_num_bits_Apost' : '6',
}
}