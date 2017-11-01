import brian2
from brian2 import pF, nS, mV, ms, pA, nA


class ExpAdaptIF_chip():
    def __init__(self):
        self.model = '''
            dImem/dt = ((Ia/Itau) * (Imem + Ith) + ((Ith / Itau) * ((Iin + Iconst) - Iahp - Itau)) - Imem * (1 + Iahp / Itau)) / (tau * (1 + (Ith / (Imem + Inoise + Io)))) : amp (unless refractory)
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
            '''
        self.threshold = '''
            Imem > Ispkthr
            '''
        self.reset = '''
            Imem = Ireset
            Ica += 30 * pA
            '''
        self.parameters = {
            #--------------------------------------------------------
            # Default equations disabled
            #--------------------------------------------------------
            "Ia": 0.5 * pA,                                # Feedback current
            "Iahp": 0.5 * pA,                                # Adaptation current
            "Inoise": 0.5 * pA,                                # Noise due to mismatch
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
            "Cmem": 0.5 * pF,
            #    "Cahp": 0.5 * pF,
            #---------------------------------------------------------
            # Positive feedback parameters
            #---------------------------------------------------------
            "Iagain": 1 * nA,
            "Iath": 20 * nA,
            "Ianorm": 1 * nA,
            #---------------------------------------------------------
            # Adaptative and Calcium parameters
            #---------------------------------------------------------
            "Ica": 0.5 * pA,
            "Itauahp": 0.5 * pA,
            "Ithahp": 0.5 * pA,
            #---------------------------------------------------------
            # Neuron parameters
            #---------------------------------------------------------
            "Ispkthr": 0.15 * nA,  # Spike threshold of excitatory neurons
            #    "Ispkthr": 6 * pA,  # Spike threshold of excitatory neurons
            "Ireset": 1 * pA,  # Reset Imem to Ireset after each spike
            "Ith": 0.5 * pA,
            "Itau": 4.3 * pA,
            "refP": 1 * ms,
            #---------------------------------------------------------
            # Noise parameters
            #---------------------------------------------------------
            "mu": 0.25 * pA,
            "sigma": 0.1 * pA
        }

        self.refractory = 'refP'
