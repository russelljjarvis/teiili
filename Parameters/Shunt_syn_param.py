from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second

parameters = {
    'Csyn': 0.100 * pF,
    'Io_syn': 0.5 * pA,
    'Ii_tau': 1. * pA,
    'Ut_syn': 25. * mV,
    'Vdd_syn': 1.8 * volt,
    'Vth_syn': 1.7 * volt,
    'baseweight_i': 7. * pA,
    'kn_syn': 0.75,
    'kp_syn': 0.66,
    'wPlast': 1,
    'Ii_th' : 1 * pA,
    'Ii_syn' : 0.5 * pA 
}
