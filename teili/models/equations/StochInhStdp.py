from brian2.units import * 
StochInhStdp = {'model':
'''
        dI_syn/dt = int(I_syn*decay_syn/mA + decay_probability_syn)*mA/second : amp (clock-driven)
        decay_probability_syn = rand() : 1 (constant over dt)
        Iin{input_number}_post = I_syn * sign(weight)                           : amp (summed)

        decay_syn = tausyn/(tausyn + dt) : 1

        weight                : 1
        w_plast               : 1
        lfsr_max_value_syn : second
        seed_syn : second
        lfsr_init_syn : second
        gain_syn              : amp
        tausyn               : second (constant)
        inh_learning_rate: 1 (constant, shared)
        variance_th: 1 (constant)
        delta_w : 1

        dApre/dt = int(Apre * decay_stdp_Apre + decay_probability_Apre)/second : 1 (clock-driven)
        dApost/dt = int(Apost * decay_stdp_Apost + decay_probability_Apost)/second : 1 (clock-driven)
        decay_probability_Apre = rand() : 1 (constant over dt)
        decay_probability_Apost = rand() : 1 (constant over dt)

        decay_stdp_Apre = taupre/(taupre + dt) : 1
        decay_stdp_Apost = taupost/(taupost + dt) : 1

        A_max: 1 (constant)
        dApre: 1 (constant)
        A_gain: 1 (constant)
        taupre : second (constant)
        taupost : second (constant)
        
         ''',
'on_pre':
'''

        I_syn += gain_syn * abs(weight) * w_plast
        I_syn = clip(I_syn, 0*mA, 15*mA)
        
        Apre += 15
        Apre = clip(Apre, 0, 15)
        delta_w  = (Apost/15 - variance_th) * inh_learning_rate
        w_plast = clip(w_plast + delta_w, 0, 31)
         ''',
'on_post':
'''
        Apost += 15
        Apost = clip(Apost, 0, 15)
        delta_w  = Apre/15 * inh_learning_rate
        w_plast = clip(w_plast + delta_w, 0, 31)

        
         
''',
'parameters':
{
'weight' : '1',
'w_plast' : '1',
'gain_syn' : '1. * mamp',
'tausyn' : '3. * msecond',
'inh_learning_rate' : '0.1',
'variance_th' : '0.12',
'taupre': '20 * msecond',
'taupost': '20 * msecond'
}
}
