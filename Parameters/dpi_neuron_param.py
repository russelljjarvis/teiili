from brian2 import pF, nS, mV, ms, pA, nA
from NCSBrian2Lib.Parameters import constants

parameters = {"kn": constants.kappa_n,
              "kp": constants.kappa_p,
              "Ut": constants.Ut,
              "Io": 0.5 * pA,
              "Cmem": 1.5 * pF,
              "Ispkthr": 1. * nA,
              "refP": 1. * ms,
              "Ireset": 0.6 * pA,
              "Iconst": 0.5 * pA,
              ##################
              "Itau": 13. * pA,  #
              "Ishunt": 0.5 * pA,  #
              "Ith": 0.9 * pA,  #
              #  ADAPTATION  #################
              "Ica": 0. * pA,
              "Itauahp": 120.0 * pA,
              "Ithahp": 20 * pA,
              "Cahp": 0.5 * pF,
              #  POSTIVE FEEDBACK #################
              "Iath": 0.5 * nA,
              "Iagain": 50. * pA,
              "Ianorm": 10. * pA,
              }
