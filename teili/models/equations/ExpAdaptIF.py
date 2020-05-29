from brian2.units import * 
ExpAdaptIF = {'model':
'''
        dVm/dt  = (Ileak + Iexp + Iin + Iconst + Inoise - Iadapt)/Cm  : volt (unless refractory)
        Ileak   : amp                            # leak current

        Iexp = gL*DeltaT*exp((Vm - VT)/DeltaT) : amp
        dIadapt/dt = -(gAdapt*(EL - Vm) + Iadapt)/tauIadapt : amp
        Inoise  : amp                            # noise current
        Iconst  : amp                            # additional input current
        Cm      : farad     (constant)           # membrane capacitance
        refP    : second    (constant)           # refractory period
        Vthr = (VT + 5 * DeltaT) : volt
        Vres    : volt      (constant)           # reset potential
        gL      : siemens   (constant)           # leak conductance
        


        tauIadapt : second  (constant) # adaptation time constant
        gAdapt    : siemens (constant) # adaptation decay parameter
        wIadapt   : amp     (constant) # adaptation weight
        EL        : volt    (constant) # reversal potential
        



        VT      : volt (constant)
        DeltaT  : volt (constant) # slope factor
        
         
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
        
        Iadapt += wIadapt;
         
          
         ''',
'parameters':
{
'Cm' : '281. * pfarad',
'refP' : '2. * msecond',
'Ileak' : '0. * amp',
'Iadapt' : '0. * amp',
'Inoise' : '0. * amp',
'Iconst' : '0. * amp',
'Vres' : '-70.6 * mvolt',
'gAdapt' : '4. * nsiemens',
'wIadapt' : '80.5 * pamp',
'tauIadapt' : '144. * msecond',
'EL' : '-70.6 * mvolt',
'gL' : '4.3 * nsiemens',
'DeltaT' : '2. * mvolt',
'VT' : '-50.4 * mvolt',
}
}