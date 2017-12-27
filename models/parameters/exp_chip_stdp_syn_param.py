from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second
from NCSBrian2Lib import constants

parameters={
        'Csyn': 1.5 * pF,
        'Io_syn': 0.5 * pA,
        'Ie_tau': 10. * pA,
        'Ii_tau': 10. * pA,
        'Q_diffAPrePost': 1.05,
        'Ut_syn': constants.UT,
        'Vdd_syn': 1.8 * volt,
        'Vth_syn': 1.7 * volt,
        'baseweight_e': 50. * nA,
        'baseweight_i': 50. * nA,
        'diffApre': 0.01,
        'kn_syn': constants.KAPPA_N,
        'kp_syn': constants.KAPPA_P,
        'taupost': 20. * ms,
        'taupre': 20. * ms,
        'w': 0,
        'wPlast': 0,
        'w_max': 0.01,
        'Ie_th' : 10 * pA,
        'Ii_th' : 10 * pA,
        'Ie_syn' : 0.5 * pA,
        'Ii_syn' : 0.5 * pA 
        }
