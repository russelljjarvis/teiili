from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second

parameters = {
    'Csyn': 1.5 * pF,
    'Io_syn': 0.5 * pA,
    'Ie_tau': 10. * pA,
    'Ii_tau': 10. * pA,
    'Ut_syn': 25. * mV,
    'baseweight_e': 50. * pA,
    'baseweight_i': 50. * pA,
    'kn_syn': 0.75,
    'kp_syn': 0.66,
    'wPlast': 1,
    'Ie_th': 10 * pA,
    'Ii_th': 10 * pA,
    'Ie_syn': 0.5 * pA,
    'Ii_syn': 0.5 * pA
}
