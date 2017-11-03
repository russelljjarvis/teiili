from NCSBrian2Lib.Parameters.Neurons.ExpAdapIF_chip_param import parameters


ExpAdapIF_chip = {'model': '''
            dImem/dt = ((Ia/Itau) * (Imem + Ith) + ((Ith / Itau) * ((Iin + Iconst) - Iahp - Itau)) - Imem * (1 + Iahp / Itau)) / (tau * (1 + (Ith / (Imem + Io)))) : amp (unless refractory)
            dIahp/dt = (-Iahp + Iahpmax) / tauahp : amp             # adaptation current
            dIca/dt = (Iahpmax-Ica) / tauca : amp
            Iahpmax = (Ica / Itauahp) * Ithahp : amp                # Ratio of currents through diffpair and adaptation block
            Ia = Iagain / (1 + exp(-(Imem - Iath) / Ianorm)) : amp  # postive feedback current
            tauahp = (Cahp * Ut) / (kappa * Itauahp) : second       # time constant of adaptation
            tau = (Cmem * Ut) / (kappa * Itau) : second             # Membrane time constant
            kappa = (kn + kp) / 2 : 1

            Inoise  : amp                                    # Noise due to mismatch
            kn      : 1 (shared, constant)                   # subthreshold slope factor for nFETs
            kp      : 1 (shared, constant)                   # subthreshold slope factor for pFETs
            Ut      : volt (shared, constant)                # Thermal voltage
            Io      : amp (shared, constant)                 # Dark current
            Cmem    : farad (shared, constant)               # Membrane capacitance
            Ispkthr : amp (constant)                         # Spiking threshold
            Ireset  : amp (shared, constant)                 # Reset current
            refP    : second    (shared, constant)           # refractory period (It is still possible to set it to False)
            Ith     : amp (constant)                         # DPI threshold (low pass filter).
            Itau    : amp (constant)                         # Leakage current
            Iconst  : amp (constant)                         # Additional input current similar to constant current injection
            Ithahp  : amp (constant)
            Itauahp : amp (constant)
            Cahp    : farad (constant)
            tauca   : second (constant)
            Iagain  : amp (shared, constant)
            Iath    : amp (shared, constant)
            Ianorm  : amp (shared, constant)
            ''',
        'threshold': '''Imem > Ispkthr''',
        'reset': '''
            Imem = Ireset
            Ica += 30 * pA
            ''',
        'parameters': parameters,
        'refractory': 'refP'}
