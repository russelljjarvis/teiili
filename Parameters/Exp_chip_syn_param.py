from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second

parameters={
        'Csyn': 0.100 * pF,
        'Io_syn': 0.5 * pA,
        'Itau_e': 1. * pA,
        'Itau_i': 1. * pA,
        'Ut_syn': 25. * mV,
        'Vdd_syn': 1.8 * volt,
        'Vth_syn': 1.7 * volt,
        'baseweight_e': 7. * pA,
        'baseweight_i': 3. * pA,
        'kernel_e': 0. * second ** -1 * pA,
        'kernel_i': 0. * second ** -1 * pA,
        'kn_syn': 0.75,
        'kp_syn': 0.66,
        'wPlast': 1
        }
