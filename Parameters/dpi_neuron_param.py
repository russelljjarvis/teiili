from brian2 import pF, nS, mV, ms, pA, nA

parameters = {"kn": 0.75,
              "kp": 0.66,
              "Ut": 25. * mV,
              "Io": 0.5 * pA,
              "Cmem": 2. * pF,
              "Ispkthr": 150. * nA,
              "refP": 1. * ms,
              "Ireset": 0.5 * pA,
              "Iconst": 0.5 * pA,
              ##################
              "Itau": 60. * pA,  #
              "Ishunt": 0.5 * pA,  #
              "Ith": 0.3 * nA,  #
              #  ADAPTATION  #################
              "Ica": 20. * nA,
              "tauca": 1. * ms,
              ##################
              "Itauahp": 10.0 * pA,
              "Ithahp": 0.5 * pA,
              "Cahp": 0.15 * pF,
              #  POSTIVE FEEDBACK #################
              "Iath": 50. * pA,
              "Iagain": 15. * pA,
              "Ianorm": 7. * pA,
              }
