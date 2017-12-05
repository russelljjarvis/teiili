from brian2 import pF, nS, mV, ms, pA, nA
from NCSBrian2Lib.Parameters import constants

parameters = {"kn": constants.kappa_n,
              "kp": constants.kappa_p,
              "Ut": constants.Ut,
              "Io": constants.Io,
              "Cmem": 1.5 * pF,
              "Ispkthr": 1. * nA,
              "refP": 1. * ms,
              "Ireset": 0.6 * pA,
              "Iconst": constants.Io,
              ##################
              "Itau": 8. * pA,  #
              "Ishunt": constants.Io,  #
              "Ith": 0.9 * pA,  #
              #  ADAPTATION  #################
              "Ica": 2. * pA,
              "Itauahp": 1 * pA,
              "Ithahp": 1 * pA,
              "Cahp": 1 * pF,
	      "Iahp":constants.Io,
              #  POSTIVE FEEDBACK #################
              "Iath": 0.5 * nA,
              "Iagain": 50. * pA,
              "Ianorm": 10. * pA,
              }
