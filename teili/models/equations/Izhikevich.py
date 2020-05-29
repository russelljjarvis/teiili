from brian2.units import * 
Izhikevich = {'model':
'''
        dVm/dt  = (Ileak + Iexp + Iin + Iconst + Inoise - Iadapt)/Cm  : volt (unless refractory)
        Ileak   : amp                            # leak current

        Iexp = k*(Vm - VR)*(Vm - VT) : amp
        dIadapt/dt = -(gAdapt*(EL - Vm) + Iadapt)/tauIadapt : amp
        Inoise  : amp                            # noise current
        Iconst  : amp                            # additional input current
        Cm      : farad     (constant)           # membrane capacitance
        refP    : second    (constant)           # refractory period
        Vthr = Vpeak : volt
        Vres = VR : volt
        gL      : siemens   (constant)           # leak conductance
        


        tauIadapt = 1.0/a            : second  # adaptation time constant
        gAdapt = b                   : siemens # adaptation decay parameter
        wIadapt = d                  : amp     # adaptation weight
        EL = VR                      : volt
        






        VT      : volt                (constant)        # V integration threshold
        Vpeak   : volt                (constant)        # V spike threshold
        VR      : volt                (constant)        # V rest
        k       : siemens * volt **-1 (constant)        # slope factor
        a       : second **-1         (constant)        # recovery time constant
        b       : siemens             (constant)        # 1/Rin
        c       : volt                (constant)        # potential reset value
        d       : amp                 (constant)        # outward minus inward currents
                                                        # activated during the spike
                                                        # and affecting the after-spike
                                                        # behavior


        
         
        x : 1 (constant) # x location on 2d grid
        y : 1 (constant) # y location on 2d grid
        
         


         Iin = Iin0  : amp # input currents

         Iin0 : amp
''',
'threshold':
'''Vm > Vthr''',
'reset':
'''
        Vm = c;
        
        Iadapt += wIadapt;
        

        Iadapt += wIadapt;
        
          
         ''',
'parameters':
{
'Cm' : '250. * pfarad',
'refP' : '2. * msecond',
'Ileak' : '0. * amp',
'Iadapt' : '0. * amp',
'Inoise' : '0. * amp',
'Iconst' : '0. * amp',
'Vpeak' : '30. * mvolt',
'VR' : '-60. * mvolt',
'VT' : '-20. * mvolt',
'a' : '10. * hertz',
'b' : '0. * siemens',
'c' : '-65. * mvolt',
'd' : '200. * pamp',
'k' : '2.5e-06 * metre ** -4 * kilogram ** -2 * second ** 6 * amp ** 3',
}
}