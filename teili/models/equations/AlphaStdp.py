from brian2.units import * 
AlphaStdp = {'model':
'''
        dI_syn/dt = (-I_syn) / tausyn + kernel: amp (clock-driven)
        Iin{input_number}_post = I_syn *  sign(weight) : amp (summed)

        kernel = s/tausyn  : amp * second **-1
        tausyn : second (constant) # synapse time constant
        w_plast : 1
        baseweight : amp (constant)     # synaptic gain
        weight : 1
        

        ds/dt = -s/tausyn   : amp (clock-driven)

        tausyn_rise : second
        
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

        I_syn += 0 * amp
        
        s += baseweight * w_plast * abs(weight)

        
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
'tausyn' : '0.5 * msecond',
'w_plast' : '0',
'baseweight' : '1. * namp',
'tausyn_rise' : '2. * msecond',
'taupre' : '10. * msecond',
'taupost' : '10. * msecond',
'w_max' : '1.0',
'dApre' : '0.1',
'Q_diffAPrePost' : '1.05',
}
}