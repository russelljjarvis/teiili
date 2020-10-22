from brian2.units import * 
StochasticSyn_decay_stoch_stdp = {'model':
'''
        dI_syn/dt = int(I_syn*decay_syn/mA + decay_probability_syn)*mA/second : amp (clock-driven)
        Iin{input_number}_post = I_syn * sign(weight)                           : amp (summed)

        decay_syn = tau_syn/(tau_syn + dt) : 1

        weight                : 1
        w_plast               : 1
        decay_probability_syn : 1
        gain_syn              : amp
        tau_syn               : second (constant)
        lfsr_num_bits_syn : 1 # Number of bits in the LFSR used
        
        dApre/dt = int(Apre * decay_stdp_Apre + decay_probability_stdp)/second : 1 (clock-driven)
        dApost/dt = int(Apost * decay_stdp_Apost + decay_probability_stdp)/second : 1 (clock-driven)

        decay_stdp_Apre = taupre/(taupre + dt) : 1
        decay_stdp_Apost = taupost/(taupost + dt) : 1

        decay_probability_stdp : 1
        w_max: 1 (constant)
        dApre: 1 (constant)
        A_gain: 1 (constant)
        taupre : second (constant)
        taupost : second (constant)
        ''',
'on_pre':
'''

        I_syn += gain_syn * weight * w_plast
        
        Apre += dApre
        w_plast = int(clip(w_plast - Apost/A_gain*int(lastspike_post!=lastspike_pre), 0, w_max))
        ''',
'on_post':
'''

        
        Apost += dApre
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
'A_gain' : '4',
'dApre' : '15',
}
}