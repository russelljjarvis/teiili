from brian2.units import * 
DPIstdgm = {'model':
'''
         
        dApre/dt = -Apre / taupre : 1 (event-driven)
        dApost/dt = -Apost / taupost : 1 (event-driven)
        gain_max: 1 (shared, constant)
        taupre : second (shared, constant)
        taupost : second (shared, constant)
        dApre : 1 (shared, constant)
        dApost : 1 (shared, constant)
        Ipred_plast : 1
        Q_diffAPrePost : 1 (shared, constant)
        scaling_factor : 1 (shared, constant)
        ''',
'on_pre':
'''

         
        Apre += dApre*gain_max
        Ipred_plast = clip(Ipred_plast + Apost, 0, gain_max)
        Ipred_post = (Ipred_post - (scaling_factor * Ipred_plast)) * (Ipred_post>0)
        ''',
'on_post':
'''

         
        Apost += -dApre * (taupre / taupost) * Q_diffAPrePost * gain_max
        Ipred_plast = clip(Ipred_plast + Apre, 0, gain_max)
        
''',
'parameters':
{
'dApre' : '0.01',
'Ipred_plast' : '0.0',
'gain_max' : '1.0',
'taupre' : '5 * msecond',
'taupost' : '5 * msecond',
'Q_diffAPrePost' : '1.05',
'scaling_factor' : '0.1',
}
}