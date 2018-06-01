
"""This file contains dictionaries of neuron equations or modules,
   combined by the neuron equation builder.

Modules:
    activity (TYPE): Description
    currentEquationsets (TYPE): Description
    currentParameters (TYPE): Description
    i_a (TYPE): Description
    i_ahp (TYPE): Description
    i_ahpPara (TYPE): Description
    i_aPara (TYPE): Description
    i_exponentialPara (TYPE): Description
    i_model_template (TYPE): Description
    i_model_templatePara (TYPE): Description
    i_noise (TYPE): Description
    i_noisePara (TYPE): Description
    i_nonLeakyPara (TYPE): Description
    modes (TYPE): Description
    none (TYPE): Description
    nonePara (dict): Description
    spatial (TYPE): Description
    v_adapt (TYPE): Description
    v_expCurrent (TYPE): Description
    v_leak (TYPE): Description
    v_model_template (TYPE): Description
    v_model_templatePara (TYPE): Description
    v_noise (TYPE): Description
    voltageEquationsets (TYPE): Description
    voltageParameters (TYPE): Description
"""

from brian2 import pF, nS, mV, ms, pA, nA

# voltage based equation building blocks
v_model_template = {'model': """
         dVm/dt  = (Ileak + Iexp + Iin + Iconst + Inoise - Iadapt)/Cm  : volt (unless refractory)
         Ileak   : amp                            # leak current
         Iexp    : amp                            # exponential current
         Iadapt  : amp                            # adaptation current
         Inoise  : amp                            # noise current
         Iconst  : amp                            # additional input current
         Cm      : farad     (shared, constant)   # membrane capacitance
         refP    : second    (shared, constant)   # refractory period (It is still possible to set it to False)
         Vthr    : volt      (shared)
         Vres    : volt      (shared, constant)   # reset potential
         """,
                    'threshold': "Vm > Vthr ",
                    'reset': "Vm = Vres; "}

v_model_templatePara = {"Cm": 281 * pF,
                        "refP": 2 * ms,
                        "Ileak": 0 * pA,
                        "Iexp": 0 * pA,
                        "Iadapt": 0 * pA,
                        "Inoise": 0 * pA,
                        "Iconst": 0 * pA,
                        "Vthr": -50.4 * mV,
                        "Vres": -70.6 * mV
                        }

# exponential current (see exponential I&F Model)
v_expCurrent = {'model': """
            #exponential
            %Iexp = gL*DeltaT*exp((Vm - VT)/DeltaT) : amp
            VT      : volt      (shared, constant)        #
            DeltaT  : volt      (shared, constant)        # slope factor

            %Vthr = (VT + 5 * DeltaT) : volt  (shared)
            """,
                'threshold': '',
                'reset': ''}
# leak
v_leak = {'model': """
          #leak
          %Ileak = -gL*(Vm - EL) : amp
          gL      : siemens   (shared, constant)        # leak conductance
          EL      : volt      (shared, constant)        # leak reversal potential
         """,
          'threshold': '',
          'reset': ''}

# adaptation
v_adapt = {'model': """
        #adapt
        %dIadapt/dt = -(gAdapt*(EL - Vm) + Iadapt)/tauIadapt : amp
        tauIadapt  : second    (shared, constant)        # adaptation time constant
        gAdapt     : siemens   (shared, constant)        # adaptation decay parameter
        wIadapt    : amp       (shared, constant)        # adaptation weight
        """,
           'threshold': '',
           'reset': 'Iadapt += wIadapt; '}
# noise
v_noise = {'model': """
        %Inoise = xi*Anoise*(second**0.5) : amp
        Anoise  : amp       (constant)
        """,
           'threshold': '',
           'reset': ''}


# independent equation building blocks

# spatial location
spatial = {'model': """
           x : 1         (constant)        # x location on 2d grid (only set it if you need it)
           y : 1         (constant)        # y location on 2d grid
           """,
           'threshold': '',
           'reset': ''}


# activity
activity = {'model': """
        dActivity/dt = -Activity/tauAct : 1
        tauAct : second (shared, constant)
        """,
            'threshold': '',
            'reset': 'Activity += 1;'}


# none
none = {'model': '',
        'threshold': '',
        'reset': ''}

# current based template
# Silicon Neuron as in Chicca et al. 2014
# Author: Moritz Milde
# Code partially adapted from Daniele Conti and Llewyn Salt
# Email: mmilde@ini.uzh.ch
i_model_template = {'model': '''
            dImem/dt = (((Ith_clip / Itau_clip) * (Iin_clip  + Ia_clip - Ishunt_clip - Iahp_clip)) - Ith_clip - ((1 + ((Ishunt_clip + Iahp_clip - Ia_clip) / Itau_clip)) * Imem)) / (tau * ((Ith_clip/(Imem + Io)) + 1)) : amp (unless refractory)

            Iahp      : amp
            Ia        : amp
            Iahp_clip : amp

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
            Itau    : amp (constant)                         # Leakage current
            Iconst  : amp (constant)                         # Additional input current similar to constant current injection
            Ishunt  : amp (constant)                         # Shunting inhibitory current (directly affects soma)
            Ica     : amp (constant)
         ''',
                    'threshold': "Imem > Ispkthr",
                    'reset': "Imem = Ireset;"}

i_model_templatePara = {
    #--------------------------------------------------------
    # Default equations disabled
    #--------------------------------------------------------
    "Inoise": 0.5 * pA,                                # Noise due to mismatch
    "Iconst": 0.5 * pA,
    #--------------------------------------------------------
    # VLSI process parameters
    #--------------------------------------------------------
    "kn": 0.75,
    "kp": 0.66,
    "Ut": 25 * mV,
    "Io": 0.5 * pA,
    #---------------------------------------------------------
    # Silicon neuron parameters
    #---------------------------------------------------------
    "Cmem": 1.5 * pF,
    #---------------------------------------------------------
    # Positive feedback parameters
    #---------------------------------------------------------
    "Ia": 0.5 * pA,                                # Feedback current
    "Iath": 0.5 * nA,
    "Iagain": 50. * pA,
    "Ianorm": 10. * pA,
    #---------------------------------------------------------
    # Adaptative and Calcium parameters
    #---------------------------------------------------------
    "Ica": 0.5 * pA,
    "Itauahp": 0.5 * pA,
    "Ithahp": 0.5 * pA,
    "Cahp": 0.5 * pF,
    "Iahp": 0.5 * pA,                                # Adaptation current
    #---------------------------------------------------------
    # Shunting inhibition
    #---------------------------------------------------------
    "Ishunt": 0.5 * pA,
    #---------------------------------------------------------
    # Neuron parameters
    #---------------------------------------------------------
    "Ispkthr": 1. * nA,  # Spike threshold of excitatory neurons
    "Ireset": 0.6 * pA,  # Reset Imem to Ireset after each spike
    "Ith": 0.9 * pA,
    "Itau": 8 * pA,
    "refP": 1 * ms
}


# noise
i_noise = {'model': """
          mu : amp
          sigma : amp
          b = sign(2 * rand() -1) : 1 (constant over dt)
          %Inoise = b * (sigma * randn() + mu) : amp (constant over dt)

         """,
           'threshold': '',
           'reset': ''}

i_noisePara = {"mu": 0.25 * pA,
               "sigma": 0.1 * pA}

# feedback
i_a = {'model': """
        %Ia = Iagain / (1 + exp(-(Imem - Iath) / Ianorm)) : amp  # postive feedback current
        %Ia_clip = Ia*(Imem>Io) + 2*Io*(Imem<=Io)    : amp
        Iagain : amp (constant)
        Iath : amp (constant)
        Ianorm : amp (constant)

         """,
       'threshold': '',
       'reset': ''}

i_aPara = {"Iath": 0.5 * nA,
           "Iagain": 50. * pA,
           "Ianorm": 10. * pA}

# adaptation
i_ahp = {'model': """
          %dIahp/dt = (- Ithahp_clip - Iahp + 2*Io*(Iahp<=Io)) / (tauahp * (Ithahp_clip / Iahp + 1)) : amp # adaptation current
          %Iahp_clip = Iahp*(Imem>Io) + Io*(Imem<=Io)  : amp
          tauahp = (Cahp * Ut) / (kappa * Itauahp) : second # time constant of adaptation
          Iahpmax = (Ica / Itauahp) * Ithahp_clip : amp     # Ratio of currents through diffpair and adaptation block
          Ithahp : amp (constant)
          Itauahp : amp (constant)
          Cahp : farad (constant)
         """,
         'threshold': '',
         'reset': '''
                  Iahp += Iahpmax;
                  '''}

i_ahpPara = {"Itauahp": 1 * pA,
             "Ithahp": 1 * pA,
             "Ica": 2 * pA,
             "Cahp": 1 * pF}


i_exponentialPara = {"Ith": 0.9 * pA,
                     "Iath": 0.5 * nA,
                     "Iagain": 50 * pA,
                     "Ianorm": 10 * pA,
                     "Itau": 8 * pA}

i_nonLeakyPara = {"Itau": 0.5 * pA}

nonePara = {}

modes = {'current': i_model_template, 'voltage': v_model_template}

currentEquationsets = {'calciumFeedback': i_ahp, 'exponential': i_a,
                       'leaky': none, 'non-leaky': none,
                       'spatial': spatial, 'gaussianNoise': i_noise, 'none': none, 'linear': none}

voltageEquationsets = {'calciumFeedback': v_adapt, 'exponential': v_expCurrent,
                       'leaky': v_leak, 'non-leaky': none,
                       'spatial': spatial, 'gaussianNoise': v_noise, 'none': none, 'linear': none}

currentParameters = {'current': i_model_templatePara, 'calciumFeedback': i_ahpPara,
                     'exponential': i_exponentialPara, 'leaky': nonePara, 'non-leaky': i_nonLeakyPara,
                     'spatial': nonePara, 'gaussianNoise': i_noisePara, 'none': nonePara, 'linear': nonePara}

voltageParameters = {'voltage': v_model_templatePara, 'calciumFeedback': nonePara,
                     'exponential': nonePara, 'leaky': nonePara, 'non-leaky': nonePara,
                     'spatial': nonePara, 'gaussianNoise': nonePara, 'none': nonePara, 'linear': nonePara}
