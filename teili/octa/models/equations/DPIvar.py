'''DPI neuron which keeps track of its Imem variance

'''
from brian2.units import *
DPIvar = {'model':
       '''
        dImem/dt = (((Ith_clip / Itau_clip) * (Iin_clip  + Ia_clip - Ishunt_clip - Iahp_clip)) - Ith_clip - ((1 + ((Ishunt_clip + Iahp_clip - Ia_clip) / Itau_clip)) * Imem)) / (tau * ((Ith_clip/(Imem + Io)) + 1)) : amp (unless refractory)

        dIahp/dt = (- Ithahp_clip - Iahp + 2*Io*(Iahp<=Io)) / (tauahp * (Ithahp_clip / Iahp + 1)) : amp # adaptation current
        Ia = Iagain / (1 + exp(-(Imem - Iath) / Ianorm)) : amp  # postive feedback current
        Iahp_clip = Iahp*(Imem>Io) + Io*(Imem<=Io)  : amp

        Itau_clip = Itau*(Imem>Io) + Io*(Imem<=Io)  : amp
        Ith_clip = Ith*(Imem>Io) + Io*(Imem<=Io)    : amp
        Iin_clip = clip(Iin+Iconst,Io, 1*amp) : amp
        Ia_clip = Ia*(Imem>Io) + 2*Io*(Imem<=Io)    : amp
        Ithahp_clip = Ithahp*(Iahp>Io) + Io*(Iahp<=Io) : amp
        Ishunt_clip = clip(Ishunt, Io, Imem) : amp

        tau = (Cmem * Ut) / (kappa * Itau_clip) : second        # Membrane time constant
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
        Itau    : amp                                    # Leakage current
        Iconst  : amp (constant)                         # Additional input current similar to constant current injection
        Ishunt  : amp (constant)                         # Shunting inhibitory current (directly affects soma)
        Ica     : amp (constant)
        adaptive_threshold : amp


        tauahp = (Cahp * Ut) / (kappa * Itauahp) : second # time constant of adaptation
        Iahpmax = (Ica / Itauahp) * Ithahp_clip : amp     # Ratio of currents through diffpair and adaptation block
        Ithahp : amp (constant)
        Itauahp : amp (constant)
        Cahp : farad (constant)



        Iagain : amp (constant)
        Iath : amp (constant)
        Ianorm : amp (constant)


        x : 1         (constant)        # x location on 2d grid (only set it if you need it)
        y : 1         (constant)        # y location on 2d grid

        activity_proxy : amp
        normalized_activity_proxy : 1


        Iin = Ie0+Ii0 : amp # input currents
        Ie0 : amp
        Ii0 : amp
''',
       'threshold':
       '''Imem > Ispkthr''',
       'reset':
       '''Imem = Ireset;
          Iahp += Iahpmax;
          Itau += adaptive_threshold;
                  ''',
       'parameters':
       {
           'Cahp': '1. * pfarad',
           'Iagain': '50. * pamp',
           'Ith': '0.9 * pamp',
           'Ireset': '0.6 * pamp',
           'refP': '1. * msecond',
           'Inoise': '0.5 * pamp',
           'Itauahp': '1. * pamp',
           'Ithahp': '1. * pamp',
           'Iconst': '0.5 * pamp',
           'Itau': '8. * pamp',
           'Ut': '25. * mvolt',
           'kn': '0.75',
           'Iath': '0.5 * namp',
           'Ispkthr': '1. * namp',
           'kp': '0.66',
           'Ianorm': '10. * pamp',
           'Ishunt': '0.5 * pamp',
           'Cmem': '1.5 * pfarad',
           'Io': '0.5 * pamp',
           'Ica': '2. * pamp',
           'Iahp': '0.5 * pamp',
           'adaptive_threshold': '0.0 * pamp',
       }
       }
