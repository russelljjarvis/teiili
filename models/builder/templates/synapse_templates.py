# -*- coding: utf-8 -*-
"""This file contains dictionaries of neuron synapse or modules,
   combined by the synapse equation builder. 

Modules:
    alphakernel (TYPE): Description
    alphaPara_conductance (TYPE): Description
    alphaPara_current (TYPE): Description
    conductance_Parameters (TYPE): Description
    conductancekernels (TYPE): Description
    current_Parameters (TYPE): Description
    currentkernels (TYPE): Description
    currentPara (TYPE): Description
    Dpi (TYPE): Description
    DPI_Parameters (TYPE): Description
    DpiPara (TYPE): Description
    fusi (TYPE): Description
    fusiPara_conductance (TYPE): Description
    fusiPara_current (TYPE): Description
    gaussiankernel (TYPE): Description
    gaussianPara_conductance (TYPE): Description
    gaussianPara_current (TYPE): Description
    modes (dict): Description
    none (dict): Description
    nonePara (dict): Description
    plasticitymodels (TYPE): Description
    resonantkernel (TYPE): Description
    resonantPara_conductance (TYPE): Description
    resonantPara_current (TYPE): Description
    reversalPara (TYPE): Description
    reversalsyn (TYPE): Description
    stdp (TYPE): Description
    stdpPara_conductance (TYPE): Description
    stdpPara_current (TYPE): Description
    template (TYPE): Description
"""
from brian2 import pF, nS, mV, ms, pA, nA, volt, second

############################################################################################
#######_____TEMPLATE MODEL AND PARAMETERS_____##############################################
############################################################################################


# none model is useful when adding exponential kernel and nonplasticity at the synapse as they already present in the template model
none = {'model': ''' ''', 'on_pre': ''' ''', 'on_post': ''' '''}


current = {'model': '''
            dIe_syn/dt = (-Ie_syn) / tausyne + kernel_e: amp (clock-driven)
            dIi_syn/dt = (-Ii_syn) / tausyni + kernel_i : amp (clock-driven)

            kernel_e : amp * second **-1
            kernel_i : amp * second **-1

            tausyne : second (constant) # synapse time constant
            tausyni : second (constant) # synapse time constant
            wPlast : 1

            baseweight_e : amp (constant)     # synaptic gain
            baseweight_i : amp (constant)     # synaptic gain
            weight : 1

            ''',

            'on_pre': '''
            Ie_syn += baseweight_e * weight *wPlast*(weight>0)
            Ii_syn += baseweight_i * weight *wPlast*(weight<0)
            ''',

            'on_post': ''' '''
            }

# standard parameters for current based models
currentPara = {"tausyne": 5 * ms,
               "tausyni": 5 * ms,
               "wPlast": 1,
               "baseweight_e": 1 * nA,
               "baseweight_i": 1 * nA,
               "kernel_e": 0 * nA * ms**-1,
               "kernel_i": 0 * nA * ms**-1
               }

# Additional equations for conductance based models
conductance = {'model': '''
               dgIe/dt = (-gIe) / tausyne + kernel_e: siemens (clock-driven)
               dgIi/dt = (-gIi) / tausyni + kernel_i : siemens (clock-driven)               

               Ie_syn = gIe*(EIe - Vm_post) :amp
               Ii_syn = gIi*(EIi - Vm_post) :amp

               EIe : volt (shared,constant)             # excitatory reversal potential
               EIi : volt (shared,constant)             # inhibitory reversal potential

               kernel_e : siemens * second **-1
               kernel_i : siemens * second **-1

               tausyne : second (constant) # synapse time constant
               tausyni : second (constant) # synapse time constant
               wPlast : 1
               
               baseweight_e : siemens (constant)     # synaptic gain
               baseweight_i : siemens (constant)     # synaptic gain
               weight : 1

               ''',
                
               'on_pre': '''
               gIe += baseweight_e * weight *wPlast*(weight>0)
               gIi += baseweight_i * weight *wPlast*(weight<0)
               ''',
               
               'on_post': ''' '''
               }

# standard parameters for conductance based models
conductancePara = {"Ige": 0 * nS,
                   "tausyne": 5 * ms,
                   # We define tausyn again here since its different from current base, is this a problem?
                   "tausyni": 6 * ms,
                   "EIe": 60.0 * mV,
                   "EIi": -90.0 * mV,
                   "wPlast": 1,
                   # should we find the way to replace baseweight_e/i, since we already defined it in template?
                   "baseweight_e": 7 * nS,
                   "baseweight_i": 3 * nS,
                   "kernel_e": 0 * nS * ms**-1,
                   "kernel_i": 0 * nS * ms**-1
                   }

# Dpi type model
Dpi = {'model': '''
        dIe_syn/dt = (-Ie_syn - Ie_gain + 2*Io_syn*(Ie_syn<=Io_syn))/(tausyne*((Ie_gain/Ie_syn)+1)) : amp (clock-driven)
        dIi_syn/dt = (-Ii_syn - Ii_gain + 2*Io_syn*(Ii_syn<=Io_syn))/(tausyni*((Ii_gain/Ii_syn)+1)) : amp (clock-driven)


        weight : 1
        wPlast : 1

        Ie_gain = Io_syn*(Ie_syn<=Io_syn) + Ie_th*(Ie_syn>Io_syn) : amp
        Ii_gain = Io_syn*(Ii_syn<=Io_syn) + Ii_th*(Ii_syn>Io_syn) : amp

        Itau_e = Io_syn*(Ie_syn<=Io_syn) + Ie_tau*(Ie_syn>Io_syn) : amp
        Itau_i = Io_syn*(Ii_syn<=Io_syn) + Ii_tau*(Ii_syn>Io_syn) : amp

        baseweight_e : amp (constant)     # synaptic gain
        baseweight_i : amp (constant)     # synaptic gain
        tausyne = Csyn * Ut_syn /(kappa_syn * Itau_e) : second
        tausyni = Csyn * Ut_syn /(kappa_syn * Itau_i) : second
        kappa_syn = (kn_syn + kp_syn) / 2 : 1


        Iw_e = weight*baseweight_e  : amp
        Iw_i = -weight*baseweight_i  : amp

        Ie_tau       : amp (constant)
        Ii_tau       : amp (constant)
        Ie_th        : amp (constant)
        Ii_th        : amp (constant)
        kn_syn       : 1 (constant)
        kp_syn       : 1 (constant)
        Ut_syn       : volt (constant)
        Io_syn       : amp (constant)
        Csyn         : farad (constant)
        ''',
       'on_pre': '''
        Ie_syn += Iw_e*wPlast*Ie_gain*(weight>0)/(Itau_e*((Ie_gain/Ie_syn)+1))
        Ii_syn += Iw_i*wPlast*Ii_gain*(weight<0)/(Itau_i*((Ii_gain/Ii_syn)+1))
        ''',
       'on_post': ''' ''',
       }

# standard parameters for Dpi models
DpiPara = {
    'Io_syn': 0.5 * pA,
    'kn_syn': 0.75,
    'kp_syn': 0.66,
    'Ut_syn': 25. * mV,
    "Igain": 15 * pA,
    'Csyn': 1.5 * pF,
    'Ie_tau': 10. * pA,
    'Ii_tau': 10. * pA,
    'Ie_th': 10 * pA,
    'Ii_th': 10 * pA,
    'Ie_syn': 0.5 * pA,
    'Ii_syn': 0.5 * pA,
    'wPlast': 1,
    'baseweight_e': 50. * pA,
    'baseweight_i': 50. * pA
}


############################################################################################
#######_____ADDITIONAL EQUATIONS BLOCKS AND PARAMETERS_____#################################
############################################################################################
# Every block must specifies additional model, pre and post spike equations, as well as
#  two different sets (dictionaries) of parameters for conductance based models or current models

# If you want to ovverride an equation add '%' before the variable of your block's explicit equation

# example:  Let's say we have the simplest model (current one with template equation),
# and you're implementing a new block with this explicit equation : d{synvar_e}/dt = (-{synvar_e})**2 / synvar_e,
# if you want to override the equation already declared in the template: d{synvar_e}/dt = (-{synvar_e}) / tausyne + kernel_e:
# your equation will be : %d{synvar_e}/dt = (-{synvar_e})**2 / synvar_e


########_____Plasticity Blocks_____#########################################################
# you need to declare two set of parameters for every block : (one for current based models and one for conductance based models)

# Fusi learning rule ##
fusi = {'model': '''
      dCa/dt = (-Ca/tau_ca) : volt (event-driven) #Calcium Potential

      updrift = 1.0*(w>theta_w) : 1
      downdrift = 1.0*(w<=theta_w) : 1

      dw/dt = (alpha*updrift)-(beta*downdrift) : 1 (event-driven) # internal weight variable

      wplus: 1 (shared)
      wminus: 1 (shared)
      theta_upl: volt (shared, constant)
      theta_uph: volt (shared, constant)
      theta_downh: volt (shared, constant)
      theta_downl: volt (shared, constant)
      theta_V: volt (shared, constant)
      alpha: 1/second (shared,constant)
      beta: 1/second (shared, constant)
      tau_ca: second (shared, constant)
      w_min: 1 (shared, constant)
      w_max: 1 (shared, constant)
      theta_w: 1 (shared, constant)
      w_ca: volt (shared, constant)     ''',

        'on_pre': '''
      up = 1. * (Vm_post>theta_V) * (Ca>theta_upl) * (Ca<theta_uph)
      down = 1. * (Vm_post<theta_V) * (Ca>theta_downl) * (Ca<theta_downh)
      w += wplus * up - wminus * down
      w = clip(w,w_min,w_max)
      wPlast = floor(w+0.5)
      ''',

        'on_post': '''Ca += w_ca'''}

fusiPara_current = {"wplus": 0.2,
                    "wminus": 0.2,
                    "theta_upl": 180 * mV,
                    "theta_uph": 1 * volt,
                    "theta_downh": 90 * mV,
                    "theta_downl": 50 * mV,
                    "theta_V": -59 * mV,
                    "alpha": 0.0001 / second,
                    "beta": 0.0001 / second,
                    "tau_ca": 8 * ms,
                    "w_ca": 250 * mV,
                    "w_min": 0,
                    "w_max": 1,
                    "theta_w": 0.5,
                    "w": 0
                    }

fusiPara_conductance = {"wplus": 0.2,
                        "wminus": 0.2,
                        "theta_upl": 180 * mV,
                        "theta_uph": 1 * volt,
                        "theta_downh": 90 * mV,
                        "theta_downl": 50 * mV,
                        "theta_V": -59 * mV,
                        "alpha": 0.0001 / second,
                        "beta": 0.0001 / second,
                        "tau_ca": 8 * ms,
                        "w_ca": 250 * mV,
                        "w_min": 0,
                        "w_max": 1,
                        "theta_w": 0.5,
                        "w": 0
                        }

# STDP learning rule ##
stdp = {'model': '''
      dApre/dt = -Apre / taupre : 1 (event-driven)
      dApost/dt = -Apost / taupost : 1 (event-driven)
      w_max: 1 (shared, constant)
      taupre : second (shared, constant)
      taupost : second (shared, constant)
      dApre : 1 (shared, constant)
      Q_diffAPrePost : 1 (shared, constant)
      ''',

        'on_pre': '''
      Apre += dApre*w_max
      wPlast = clip(wPlast + Apost, 0, w_max) ''',

        'on_post': '''
      Apost += -dApre * (taupre / taupost) * Q_diffAPrePost * w_max
      wPlast = clip(wPlast + Apre, 0, w_max) '''}

stdpPara_current = {"baseweight_e": 7 * pA,  # should we find the way to replace since we would define it twice
                    "baseweight_i": 7 * pA,
                    "taupre": 10 * ms,
                    "taupost": 10 * ms,
                    "w_max": 1.,
                    "dApre": 0.1,
                    "Q_diffAPrePost": 1.05,
                    "wPlast": 0}

stdpPara_conductance = {"baseweight_e": 7 * nS,  # should we find the way to replace since we would define it twice
                        "baseweight_i": 3 * nS,
                        "taupre": 20 * ms,
                        "taupost": 20 * ms,
                        "w_max": 0.01,
                        "diffApre": 0.01,
                        "Q_diffAPrePost": 1.05,
                        "wPlast": 0}

########_____Kernels Blocks_____#########################################################
# you need to declare two set of parameters for every block : (one for current based models and one for conductance based models)

# TODO: THESE KERNELS ARE WRONG!

# Alpha kernel ##

alphakernel = {'model': '''
             %kernel_e = baseweight_e*(weight>0)*wPlast*weight*exp(1-t_spike/tausyne_rise)/tausyne : {unit} * second **-1
             %kernel_i = baseweight_i*(weight<0)*wPlast*weight*exp(1-t_spike/tausyni_rise)/tausyni : {unit} * second **-1
             dt_spike/dt = 1 : second (clock-driven)
             tausyne_rise : second
             tausyni_rise : second
             ''',

               'on_pre': '''

             t_spike = 0 * ms
             ''',

               'on_post': ''' '''}

alphaPara_current = {"tausyne": 2 * ms,
                     "tausyni": 2 * ms,
                     "tausyne_rise": 0.5 * ms,
                     "tausyni_rise": 0.5 * ms}

alphaPara_conductance = {"tausyne": 2 * ms,
                         "tausyni": 2 * ms,
                         "tausyne_rise": 1 * ms,
                         "tausyni_rise": 1 * ms}

# Resonant kernel ##
resonantkernel = {'model': '''
                omega: 1/second
                sigma_gaussian : second
                %kernel_e  = baseweight_e*(weight>0)*wPlast*(weight*exp(-t_spike/tausyne_rise)*cos(omega*t_spike))/tausyne : {unit} * second **-1
                %kernel_i  = baseweight_i*(weight<0)*wPlast*(weight*exp(-t_spike/tausyni_rise)*cos(omega*t_spike))/tausyni : {unit} * second **-1
                dt_spike/dt = 1 : second (clock-driven)
                tausyne_rise : second
                tausyni_rise : second
                ''',

                  'on_pre': '''

                t_spike = 0 * ms
                ''',

                  'on_post': ''' '''}

resonantPara_current = {"tausyne": 2 * ms,
                        "tausyni": 2 * ms,
                        "omega": 7 / ms,
                        "tausyne_rise": 0.5 * ms,
                        "tausyni_rise": 0.5 * ms}

resonantPara_conductance = {"tausyne": 2 * ms,
                            "tausyni": 2 * ms,
                            "omega": 1 / ms}


#  Gaussian kernel ##


gaussiankernel = {'model': '''
                  %tausyne = (sigma_gaussian_e**2)/t_spike : second
                  %tausyni = (sigma_gaussian_i**2)/t_spike : second
                  sigma_gaussian_e : second
                  sigma_gaussian_i : second

                  dt_spike/dt = 1 : second (clock-driven)
                  ''',
                  # this time we need to add this pre eq to the template pe eq

                  'on_pre': '''t_spike = 0 * ms''',

                  'on_post': ''' '''}

gaussianPara_current = {"sigma_gaussian_e": 6 * ms,
                        "sigma_gaussian_i": 6 * ms}

gaussianPara_conductance = {"sigma_gaussian_e": 6 * ms,
                            "sigma_gaussian_i": 6 * ms}


nonePara = {}


########_____Dictionary of keywords_____#########################################################
# These dictionaries contains keyword and models and parameters names useful for the __init__ subroutine
# Every new block dictionaries must be added to these definitions

modes = {'current': current, 'conductance': conductance, 'DPI': Dpi}

kernels = {'exponential': none, 'alpha': alphakernel,
                      'resonant': resonantkernel, 'gaussian': gaussiankernel}

plasticitymodels = {'nonplastic': none, 'fusi': fusi, 'stdp': stdp}



###parameters dictionaries
current_Parameters = {'current': currentPara, 'nonplastic': nonePara, 'fusi': fusiPara_current,
                      'stdp': stdpPara_current, 'exponential': nonePara, 'alpha': alphaPara_current,
                      'resonant': resonantPara_current, 'gaussian': gaussianPara_current}

conductance_Parameters = {'conductance': conductancePara, 'nonplastic': nonePara, 'fusi': fusiPara_conductance,
                          'stdp': stdpPara_conductance, 'exponential': nonePara, 'alpha': alphaPara_conductance,
                          'resonant': resonantPara_conductance, 'gaussian': gaussianPara_conductance}

DPI_Parameters = {'DPI': DpiPara, 'nonplastic': nonePara, 'fusi': fusiPara_current,
                  'stdp': stdpPara_current}
