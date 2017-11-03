from brian2 import pF, nS, mV, ms, pA, nA

parameters = {"kn": 0.75,
              "kp": 0.66,
              "Ut": 25. * mV,
              "Io": 0.5 * pA,
              "Cmem": 0.5 * pF,
              "Ica": 0.5 * pA,
              "Ispkthr": 150. * pA,
              "Ireset": 1. * pA,
              "refP": 1. * ms,
              "tauca": 20. * ms,
              "Itauahp": 0.5 * pA,
              "Ithahp": 0.5 * pA,
              "Cahp": 0.5 * pF,
              "Ith": 0.5 * pA,
              "Iath": 20. * nA,
              'Iahp': 0.5 * pA,
              "Iagain": 1. * nA,
              "Ianorm": 1. * nA,
              "Itau": 4.3 * pA,
              "mu": 0.25 * pA,
              "sigma": 0.1 * pA
              }
