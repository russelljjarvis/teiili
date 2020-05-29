from brian2.units import * 
ExpLIF = {'model':
'''
        dVm/dt  = (Ileak + Iexp + Iin + Iconst + Inoise - Iadapt)/Cm  : volt (unless refractory)
        Ileak = -gL*(Vm - EL) : amp

        Iexp = gL*DeltaT*exp((Vm - VT)/DeltaT) : amp
        Iadapt  : amp                            # adaptation current
        Inoise  : amp                            # noise current
        Iconst  : amp                            # additional input current
        Cm      : farad     (constant)           # membrane capacitance
        refP    : second    (constant)           # refractory period
        Vthr = (VT + 5 * DeltaT) : volt
        Vres    : volt      (constant)           # reset potential
        gL      : siemens   (constant)           # leak conductance
        
         



        VT      : volt (constant)
        DeltaT  : volt (constant) # slope factor
        


        EL : volt (constant) # leak reversal potential
        
        x : 1 (constant) # x location on 2d grid
        y : 1 (constant) # y location on 2d grid
        
         


         Iin = Iin0  : amp # input currents

         Iin0 : amp
''',
'threshold':
'''Vm > Vthr''',
'reset':
'''
        Vm = Vres;
        
          
     
         ''',
'parameters':
{
'Cm' : '281. * pfarad',
'refP' : '2. * msecond',
'Iadapt' : '0. * amp',
'Inoise' : '0. * amp',
'Iconst' : '0. * amp',
'Vres' : '-70.6 * mvolt',
'gL' : '4.3 * nsiemens',
'DeltaT' : '2. * mvolt',
'VT' : '-50.4 * mvolt',
'EL' : '-55. * mvolt',
}
}