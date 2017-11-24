from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second

parameters = {
    'Csyn': 0.5 * pF,
    'Io_syn': 0.5 * pA,
    'Ie_tau': 7. * pA,
    'Ii_tau': 7. * pA,
    'Ut_syn': 25. * mV,
    'baseweight_e': 7. * pA,
    'baseweight_i': 7. * pA,
    'kn_syn': 0.75,
    'kp_syn': 0.66,
    'wPlast': 1,
    "Igain": 15 * pA,
    'Ie_th': 10 * pA,
    'Ii_th': 10 * pA,
    'Ie_syn': 0.5 * pA,
    'Ii_syn': 0.5 * pA
}
