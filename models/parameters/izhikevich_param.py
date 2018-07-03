from brian2 import pF, nS, mV, ms, pA, nA, psiemens
pS = psiemens

parameters = {"Cm": 250 * pF,
              "refP": 1 * ms,
              "Ileak": 0 * pA,
              "Inoise": 0 * pA,
              "Iconst": 0 * pA,
              "VT": -20.0 * mV,
              "VR": -60.0 * mV,
              "a": 0.01 * ms**-1, # 1.0/tauIadapt
              "b": 0.0 * pS, # gAdapt
              "c": -65.0 * mV,  # Vreset
              "d": 200.0 * pA, # wIadapt
              "k": 2.5 * pS / mV 
              }
