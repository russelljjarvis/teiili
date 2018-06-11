
"""This file contains dictionaries of neuron equations or modules,
   combined by the neuron equation builder.

Modules:
    activity (TYPE): Description
    currentEquationsets (TYPE): Description
    currentParameters (TYPE): Description
    i_a (TYPE): Description
    i_ahp (TYPE): Description
    i_ahp_params (TYPE): Description
    i_a_params (TYPE): Description
    i_exponential_params (TYPE): Description
    i_model_template (TYPE): Description
    i_model_template_params (TYPE): Description
    i_noise (TYPE): Description
    i_noise_params (TYPE): Description
    i_non_leaky_params (TYPE): Description
    modes (TYPE): Description
    none (TYPE): Description
    none_params (dict): Description
    spatial (TYPE): Description
    v_adapt (TYPE): Description
    v_exp_current (TYPE): Description
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
                    'threshold': "Vm > Vthr",
                    'reset': "Vm = Vres;"}

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
#exponential
v_exp_current = {'model': """
            %Iexp = gL*DeltaT*exp((Vm - VT)/DeltaT) : amp
            VT      : volt      (shared, constant)        #
            DeltaT  : volt      (shared, constant)        # slope factor
            gL      : siemens   (shared, constant)        # leak conductance
            %Vthr = (VT + 5 * DeltaT) : volt  (shared)
            """,
                'threshold': '',
                'reset': ''}

v_exp_current_params = {"gL" : 4.3 * nS,
                       "DeltaT": 2 * mV,
                       "VT": -50.4 * mV
                       }

# quadratic current (see Izhikevich Model)
v_quad_current = {'model': """
            #quadratic
            %Iexp = k*(Vm - VR)*(Vm - VT) : amp
            %tauIadapt = 1.0/a  : second    (shared)        # adaptation time constant
            %gAdapt = b         : siemens   (shared)        # adaptation decay parameter
            %wIadapt = d         : amp      (shared)        # adaptation weight
            %EL = VR : volt
            VT      : volt                (shared, constant)        # V threshold
            VR      : volt                (shared, constant)        # V rest
            k       : siemens * volt **-1 (shared, constant)        # slope factor
            a       : second **-1         (shared, constant)        # recovery time constant
            b       : siemens             (shared, constant)        # 1/Rin
            c       : volt                (shared, constant)        # potential reset value
            d       : amp                 (shared, constant)        # outward minus inward currents
                                                                    # activated during the spike
                                                                    # and affecting the after-spike
                                                                    # behavior
            %Vthr = VT : volt  (shared)
            %Vres = VR : volt  (shared)
            """,
                  'threshold': '',
                  'reset': "%Vm = c; Iadapt += wIadapt;"}

v_quad_params = {
    "Cm": 250.0 * pF,
    "VR": -60.0 * mV,
    "VT": -20.0 * mV,
    "a": 0.01 / ms, # Nicola&Clopath2017
    "b": 0.0 * nS, # Nicola&Clopath2017
    "c": -65 * mV, # Nicola&Clopath2017
    "d": 200 * pA,  # Nicola&Clopath2017
    "k":  2.5  * nS / mV} # k = 1/Rin Nicola&Clopath2017


# leak
v_leak = {'model': """
          %Ileak = -gL*(Vm - EL) : amp
          gL      : siemens   (shared, constant)        # leak conductance
          EL      : volt      (shared, constant)        # leak reversal potential
         """,
          'threshold': '',
          'reset': ''}

v_leak_params = {"gL" : 4.3 * nS,
                 "EL" : -55 * mV
                 }

# adaptation
v_adapt = {'model': """
        %dIadapt/dt = -(gAdapt*(EL - Vm) + Iadapt)/tauIadapt : amp
        tauIadapt  : second    (shared, constant)        # adaptation time constant
        gAdapt     : siemens   (shared, constant)        # adaptation decay parameter
        wIadapt    : amp       (shared, constant)        # adaptation weight
        EL      : volt      (shared, constant)        # reversal potential
        """,
           'threshold': '',
           'reset': 'Iadapt += wIadapt;'}


v_adapt_params = {"gAdapt": 4 * nS,
                  "wIadapt": 0.0805 * nA,
                  "tauIadapt": 144 * ms,
                  "EL": -70.6 * mV
                  }


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

i_model_template_params = {
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

i_noise_params = {"mu": 0.25 * pA,
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

i_a_params = {"Iath": 0.5 * nA,
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

i_ahp_params = {"Itauahp": 1 * pA,
             "Ithahp": 1 * pA,
             "Ica": 2 * pA,
             "Cahp": 1 * pF}


i_exponential_params = {"Ith": 0.9 * pA,
                     "Iath": 0.5 * nA,
                     "Iagain": 50 * pA,
                     "Ianorm": 10 * pA,
                     "Itau": 8 * pA}

i_non_leaky_params = {"Itau": 0.5 * pA}

none_params = {}

modes = {'current': i_model_template, 'voltage': v_model_template}

current_equation_sets = {'calcium_feedback': i_ahp, 'exponential': i_a,
                         'leaky': none, 'non_leaky': none, 'quadratic': none,
                         'spatial': spatial, 'gaussian_noise': i_noise, 'none': none, 'linear': none}

voltage_equation_sets = {'calcium_feedback': v_adapt, 'exponential': v_exp_current,
                         'quadratic': v_quad_current,
                         'leaky': v_leak, 'non_leaky': none,
                         'spatial': spatial, 'gaussian_noise': v_noise, 'none': none, 'linear': none}

current_parameters = {'current': i_model_template_params, 'calcium_feedback': i_ahp_params,
                      'quadratic': none_params,
                      'exponential': i_exponential_params, 'leaky': none_params, 'non_leaky': i_non_leaky_params,
                      'spatial': none_params, 'gaussian_noise': i_noise_params, 'none': none_params, 'linear': none_params}

voltage_parameters = {'voltage': v_model_templatePara, 'calcium_feedback': v_adapt_params,
                      'exponential': v_exp_current_params, 'quadratic': v_quad_params,
                      'leaky': v_leak_params, 'non_leaky': none_params,
                      'spatial': none_params, 'gaussian_noise': none_params, 'none': none_params, 'linear': none_params}
