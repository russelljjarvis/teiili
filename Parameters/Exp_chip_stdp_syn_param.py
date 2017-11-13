from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second

parameters={
        'Csyn': 0.100 * pF,
        'Io_syn': 0.5 * pA,
        'Itau_e': 1. * pA,
        'Itau_i': 1. * pA,
        'Q_diffAPrePost': 1.05,
        'Ut_syn': 25. * mV,
        'Vdd_syn': 1.8 * volt,
        'Vth_syn': 1.7 * volt,
        'baseweight_e': 7. * nA,
        'baseweight_i': 7. * nA,
        'diffApre': 0.01,
        'kernel_e': 0. * second ** -1 * pA,
        'kernel_i': 0. * second ** -1 * pA,
        'kn_syn': 0.75,
        'kp_syn': 0.66,
        'taupost': 20. * ms,
        'taupre': 20. * ms,
        'w': 0,
        'wPlast': 0,
        'w_max': 0.01,
	"Igain" : 100 * pA 
        }
