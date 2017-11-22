from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second

parameters={
        'Csyn': 0.100 * pF,
        'Io_syn': 0.5 * pA,
        'Ie_tau': 1. * pA,
        'Ii_tau': 1. * pA,
        'Q_diffAPrePost': 1.05,
        'Ut_syn': 25. * mV,
        'Vdd_syn': 1.8 * volt,
        'Vth_syn': 1.7 * volt,
        'baseweight_e': 7. * nA,
        'baseweight_i': 7. * nA,
        'diffApre': 0.01,
        'kn_syn': 0.75,
        'kp_syn': 0.66,
        'taupost': 20. * ms,
        'taupre': 20. * ms,
        'w': 0,
        'wPlast': 0,
        'w_max': 0.01,
        'Ie_th' : 1 * pA,
        'Ii_th' : 1 * pA,
        'Ie_syn' : 0.5 * pA,
        'Ii_syn' : 0.5 * pA 
        }
