from brian2.units import * 
StochStdStdp = {'model':
'''
        dI_syn/dt = int(I_syn*decay_syn/mA + decay_probability_syn)*mA/second : amp (clock-driven)
        Iin{input_number}_post = I_syn * sign(weight)                           : amp (summed)

        decay_syn = tausyn/(tausyn + dt) : 1

        weight                : 1
        w_plast               : 1
        decay_probability_syn : 1
        gain_syn              : amp
        tausyn               : second (constant)
        
        dApre/dt = (Apre * decay_stdp_Apre)/second : 1 (clock-driven)
        dApost/dt = (Apost * decay_stdp_Apost)/second : 1 (clock-driven)

        decay_stdp_Apre = taupre/(taupre + dt) : 1
        decay_stdp_Apost = taupost/(taupost + dt) : 1

        w_max: 1 (constant)
        dApre: 1 (constant)
        taupre : second (constant)
        taupost : second (constant)
        ''',
'on_pre':
'''

        I_syn += gain_syn * weight * w_plast
        
        Apre += dApre
        w_plast = (clip(w_plast - Apost*int(lastspike_post!=lastspike_pre), 0, w_max))
        ''',
'on_post':
'''

        
        Apost += dApre
        w_plast = (clip(w_plast + Apre*int(lastspike_post!=lastspike_pre), 0, w_max))
        
''',
'parameters':
{
'weight' : '1',
'w_plast' : '1',
'gain_syn' : '1. * mamp',
'tausyn' : '3. * msecond',
'taupre' : '3. * msecond',
'taupost' : '3. * msecond',
'w_max' : '15',
'dApre' : '0.03',
}
}
