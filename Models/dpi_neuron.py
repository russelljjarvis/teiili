from NCSBrian2Lib.Parameters.dpi_neuron_param import parameters


dpi_neuron_eq = {'model': '''
                        dImem/dt = (((Ith_clip / Itau_clip) * (Iin_clip  + Ia_clip - Ishunt - Iahp_clip)) - Ith_clip - ((1 + ((Ishunt + Iahp_clip - Ia_clip) / Itau_clip)) * Imem)   ) / (tau * ((Ith_clip/(Imem + Io)) + 1)) : amp (unless refractory)

                        dIahp/dt = (- Ithahp_clip - Iahp + 2*Io*(Iahp<=Io)) / (tauahp * (Ithahp_clip / Iahp + 1)) : amp # adaptation current

                        Itau_clip = Itau*(Imem>Io) + Io*(Imem<=Io)  : amp
                        Ith_clip = Ith*(Imem>Io) + Io*(Imem<=Io)    : amp
                        Iin_clip = clip(Iin+Iconst,Io, 1*amp) : amp
                        Iahp_clip = Iahp*(Imem>Io) + Io*(Imem<=Io)  : amp
                        Ia_clip = Ia*(Imem>Io) + 2*Io*(Imem<=Io)    : amp
                        Ithahp_clip = Ithahp*(Iahp>Io) + Io*(Iahp<=Io) : amp

                        Iahpmax = (Ica / Itauahp) * Ithahp_clip : amp                # Ratio of currents through diffpair and adaptation block
                        Ia = Iagain / (1 + exp(-(Imem - Iath) / Ianorm)) : amp  # postive feedback current

                        tauahp = (Cahp * Ut) / (kappa * Itauahp) : second       # time constant of adaptation
                        tau = (Cmem * Ut) / (kappa * Itau_clip) : second        # Membrane time constant
                        kappa = (kn + kp) / 2 : 1

                        Inoise  : amp                                    # Noise due to mismatch
                        kn      : 1 (shared, constant)                   # subthreshold slope factor for nFETs
                        kp      : 1 (shared, constant)                   # subthreshold slope factor for pFETs
                        Ut      : volt (shared, constant)                # Thermal voltage
                        Io      : amp (constant)                         # Dark current
                        Cmem    : farad (shared, constant)               # Membrane capacitance
                        Ispkthr : amp (constant)                         # Spiking threshold
                        Ireset  : amp (constant)                         # Reset current
                        refP    : second    (shared, constant)           # refractory period (It is still possible to set it to False)
                        Ith     : amp (constant)                         # DPI threshold (low pass filter).
                        Itau    : amp (constant)                         # Leakage current
                        Iconst  : amp (constant)                         # Additional input current similar to constant current injection
                        Ithahp  : amp (constant)
                        Itauahp : amp (constant)
                        Cahp    : farad (constant)
                        tauca   : second (constant)
                        Iagain  : amp (constant)
                        Iath    : amp (constant)
                        Ianorm  : amp (constant)
                        Ishunt  : amp (constant)
                        Ica     : amp (constant)
                        ''',
                 'threshold': '''Imem > Ispkthr''',
                 'reset': '''
                        Imem = Ireset
                        Iahp += Iahpmax
                        ''',
                 'refractory': 'refP',
                 'method': 'euler',
                 'parameters': parameters}
