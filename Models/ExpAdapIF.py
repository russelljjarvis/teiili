from NCSBrian2Lib.Parameters.ExpAdapIF_param import parameters


ExpAdapIF_chip = {'model': '''
            dVm/dt  = (Ileak + Iexp + Iin + Iconst + Inoise - Iadapt)/Cm  : volt (unless refractory)
            Inoise  : amp                             # noise current
            Iconst  : amp                             # additional input current
            Cm                 : farad     (shared, constant)    # membrane capacitance
            refP               : second    (shared, constant)    # refractory period (It is still possible to set it to False)
            Vres               : volt      (shared, constant)        # reset potential


            #adapt
            dIadapt/dt = -(gAdapt*(EL - Vm) + Iadapt)/tauIadapt : amp
            tauIadapt  : second    (shared, constant)        # adaptation time constant
            gAdapt     : siemens   (shared, constant)        # adaptation decay parameter
            wIadapt    : amp       (shared, constant)        # adaptation weight

            #exponential
            Iexp = gL*DeltaT*exp((Vm - VT)/DeltaT) : amp
            VT      : volt      (shared, constant)        #
            DeltaT  : volt      (shared, constant)        # slope factor
            Vthr = (VT + 5 * DeltaT) : volt  (shared)

            #leak
            Ileak = -gL*(Vm - EL) : amp
            gL      : siemens   (shared, constant)        # leak conductance
            EL      : volt      (shared, constant)        # leak reversal potential
            ''',
        'threshold': '''Vm > Vthr ''',
        'reset': '''
            Vm = Vres
            Iadapt += wIadapt
            ''',
        'parameters': parameters,
        'refractory': 'refP'}
