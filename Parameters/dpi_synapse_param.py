from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second

parameters = {
    'Io_syn': 0.5 * pA,
    'kn_syn': 0.75,
    'kp_syn': 0.66,
    'Ut_syn': 25. * mV,
    'Csyn': 1. * pF,
    "Igain": 15 * pA,
    'Ie_tau': 4. * pA,
    'Ii_tau': 4. * pA,
    'Ie_th': 10 * pA,
    'Ii_th': 10 * pA,
    'Ie_syn': 0.5 * pA,
    'Ii_syn': 0.5 * pA,
    'wPlast': 1,
    'baseweight_e': 50. * pA,
    'baseweight_i': 50. * pA
}
