from brian2 import pF, nS, mV, ms, pA, nA

parameters = {"kn": 0.75,  #
              "kp": 0.66,  #
              "Ut": 25. * mV,  #
              "Io": 0.5 * pA,  #
              "Cmem": 2. * pF,  #
              "Ica": 50. * nA,
              "Ispkthr": 150. * nA,
              "Ireset": 0.5 * pA,
              "refP": 1. * ms,  #
              "tauca": 0.5 * ms,
              "Itauahp": 20.5 * pA,  #
              "Ithahp": 0.5 * pA,  #
              "Cahp": 0.15 * pF,  #
              "Ith": 40 * nA,  #
              "Iath": 50. * pA,
              "Iagain": 15. * pA,
              "Ianorm": 7. * pA,
              "Itau": 30. * pA,  #
              "Ishunt": 0.5 * pA,  #
              "Iconst": 0.5 * pA,  #
              "mu": 0.25 * pA,  #
              "sigma": 0.1 * pA  #
              }
