from brian2.units import * 
ExponentialStdp = {'model':
'''
        dI_syn/dt = (-I_syn) / tausyn + kernel: amp (clock-driven)
        Iin{input_number}_post = I_syn *  sign(weight) : amp (summed)

        kernel : amp * second **-1
        tausyn : second (constant) # synapse time constant
        w_plast : 1
        baseweight : amp (constant)     # synaptic gain
        weight : 1
        
         
        dApre/dt = -Apre / taupre : 1 (event-driven)
        dApost/dt = -Apost / taupost : 1 (event-driven)
        w_max: 1 (constant)
        taupre : second (constant)
        taupost : second (constant)
        dApre : 1 (constant)
        Q_diffAPrePost : 1 (constant)
        ''',
'on_pre':
'''

        I_syn += baseweight * abs(weight) * w_plast
        
         
        Apre += dApre*w_max
        w_plast = clip(w_plast + Apost, 0, w_max)
        ''',
'on_post':
'''
 
         
        Apost += -dApre * (taupre / taupost) * Q_diffAPrePost * w_max
        w_plast = clip(w_plast + Apre, 0, w_max)
        
''',
'parameters':
{
'tausyn' : '5. * msecond',
'w_plast' : '0',
'baseweight' : '1. * namp',
'kernel' : '0. * second ** -1 * amp',
'taupre' : '10. * msecond',
'taupost' : '10. * msecond',
'w_max' : '1.0',
'dApre' : '0.1',
'Q_diffAPrePost' : '1.05',
}
}