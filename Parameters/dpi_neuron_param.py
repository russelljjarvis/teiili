from brian2 import pF, nS, mV, ms, pA, nA

parameters = {"kn": 0.75,
              "kp": 0.66,
              "Ut": 25. * mV,
              "Io": 0.5 * pA,
              "Cmem": 2. * pF,
              "Ispkthr": 150. * pA,
              "refP": 1. * ms,
              "Ireset": 0.6 * pA,
              "Iconst": 0.5 * pA,
              ##################
              "Itau": 8. * pA,  #
              "Ishunt": 0.5 * pA,  #
              "Ith": 0.9 * pA,  #
              #  ADAPTATION  #################
              "Ica": 30. * pA,
              "tauca": 20. * ms,
              ##################
              "Itauahp": 120.0 * pA,
              "Ithahp": 20 * pA,
              "Cahp": 0.5 * pF,
              #  POSTIVE FEEDBACK #################
              "Iath": 20. * nA,
              "Iagain": 20. * nA,
              "Ianorm": 1. * nA,
              }
