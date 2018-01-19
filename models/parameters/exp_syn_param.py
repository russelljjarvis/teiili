from brian2 import ms, mV, pA, nS, nA, pF, us, volt, second


parameters={
        'baseweight_e': 1. * nA,
        'baseweight_i': 1. * nA,
        'kernel_e': 0. * second ** -1 * pA,
        'kernel_i': 0. * second ** -1 * pA,
        'tausyne': 5. * ms,
        'tausyni': 5. * ms,
        'wPlast': 1
        }
