# -*- coding: utf-8 -*-
"""This file contains dictionaries of neuron synapse or modules,
   combined by the synapse equation builder.

Modules:
    alpha_kernel (TYPE): Description
    alpha_params_conductance (TYPE): Description
    alpha_params_current (TYPE): Description
    conductance_Parameters (TYPE): Description
    conductancekernels (TYPE): Description
    current_Parameters (TYPE): Description
    currentkernels (TYPE): Description
    current_params (TYPE): Description
    Dpi (TYPE): Description
    DPI_Parameters (TYPE): Description
    dpi_params (TYPE): Description
    fusi (TYPE): Description
    fusi_params_conductance (TYPE): Description
    fusi_params_current (TYPE): Description
    gaussian_kernel (TYPE): Description
    gaussian_params_conductance (TYPE): Description
    gaussian_params_current (TYPE): Description
    modes (dict): Description
    none (dict): Description
    none_params (dict): Description
    plasticitymodels (TYPE): Description
    resonant_kernel (TYPE): Description
    resonant_params_conductance (TYPE): Description
    resonant_params_current (TYPE): Description
    reversalPara (TYPE): Description
    reversalsyn (TYPE): Description
    stdp (TYPE): Description
    stdp_para_conductance (TYPE): Description
    stdp_para_current (TYPE): Description
    template (TYPE): Description

Attributes:
    alpha_kernel (TYPE): Description
    alpha_params_conductance (TYPE): Description
    alpha_params_current (TYPE): Description
    conductance (TYPE): Description
    conductance_parameters (TYPE): Description
    conductance_params (TYPE): Description
    current (TYPE): Description
    current_parameters (TYPE): Description
    current_params (TYPE): Description
    dpi (TYPE): Description
    DPI_parameters (TYPE): Description
    dpi_params (TYPE): Description
    dpi_shunt (TYPE): Description
    DPI_shunt_parameters (TYPE): Description
    dpi_shunt_params (TYPE): Description
    fusi (TYPE): Description
    fusi_params_conductance (TYPE): Description
    fusi_params_current (TYPE): Description
    gaussian_kernel (TYPE): Description
    gaussian_params_conductance (TYPE): Description
    gaussian_params_current (TYPE): Description
    kernels (TYPE): Description
    modes (TYPE): Description
    none (dict): Description
    none_params (dict): Description
    plasticity_models (TYPE): Description
    resonant_kernel (TYPE): Description
    resonant_params_conductance (TYPE): Description
    resonant_params_current (TYPE): Description
    stdp (TYPE): Description
    stdp_para_conductance (TYPE): Description
    stdp_para_current (TYPE): Description
"""
# @Author: Moritz Milde
# @Date:   2018-06-01 11:57:02


from brian2 import pF, nS, mV, ms, pA, nA, volt, second

none = {'model': ''' ''', 'on_pre': ''' ''', 'on_post': ''' '''}

current = {'model': '''
            kernel_e : amp * second **-1
            kernel_i : amp * second **-1

            dIe_syn/dt = (-Ie_syn) / tausyne + kernel_e: amp (clock-driven)
            dIi_syn/dt = (-Ii_syn) / tausyni + kernel_i : amp (clock-driven)

            Ie{input_number}_post = Ie_syn : amp (summed)
            Ii{input_number}_post = -Ii_syn : amp (summed)

            tausyne : second (constant) # synapse time constant
            tausyni : second (constant) # synapse time constant
            w_plast : 1

            baseweight_e : amp (constant)     # synaptic gain
            baseweight_i : amp (constant)     # synaptic gain
            weight : 1

            ''',

           'on_pre': '''
            Ie_syn += baseweight_e * weight *w_plast*(weight>0)
            Ii_syn += baseweight_i * weight *w_plast*(weight<0)
            ''',

           'on_post': ''' '''
           }

# standard parameters for current based models
current_params = {"tausyne": 5 * ms,
                  "tausyni": 5 * ms,
                  "w_plast": 1,
                  "baseweight_e": 1 * nA,
                  "baseweight_i": 1 * nA,
                  "kernel_e": 0 * nA * ms**-1,
                  "kernel_i": 0 * nA * ms**-1
                  }

# Additional equations for conductance based models
conductance = {'model': '''
               dgIe/dt = (-gIe) / tausyne + kernel_e : siemens (clock-driven)
               dgIi/dt = (-gIi) / tausyni + kernel_i : siemens (clock-driven)

               Ie_syn = gIe*(EIe - Vm_post) : amp
               Ii_syn = gIi*(EIi - Vm_post) : amp

               Ie{input_number}_post = Ie_syn : amp (summed)
               Ii{input_number}_post = -Ii_syn : amp (summed)

               EIe : volt (constant)             # excitatory reversal potential
               EIi : volt (constant)             # inhibitory reversal potential

               kernel_e : siemens * second **-1
               kernel_i : siemens * second **-1

               tausyne : second (constant) # synapse time constant
               tausyni : second (constant) # synapse time constant
               w_plast : 1

               baseweight_e : siemens (constant)     # synaptic gain
               baseweight_i : siemens (constant)     # synaptic gain
               weight : 1

               ''',

               'on_pre': '''
               gIe += baseweight_e * weight *w_plast*(weight>0)
               gIi += baseweight_i * weight *w_plast*(weight<0)
               ''',

               'on_post': ''' '''
               }

# standard parameters for conductance based models
conductance_params = {"gIe": 0 * nS,
                      "tausyne": 5 * ms,
                      # We define tausyn again here since it's different from
                      # current base, is this a problem?
                      "tausyni": 6 * ms,
                      "EIe": 60.0 * mV,
                      "EIi": -90.0 * mV,
                      "w_plast": 1,
                      # should we find a way to replace baseweight_e/i, since we
                      # already defined it in template?
                      "baseweight_e": 7 * nS,
                      "baseweight_i": 3 * nS,
                      "kernel_e": 0 * nS * ms**-1,
                      "kernel_i": 0 * nS * ms**-1
                      }

# DPI type model
dpi = {'model': '''
        dIe_syn/dt = (-Ie_syn - Ie_gain + 2*Io_syn*(Ie_syn<=Io_syn))/(tausyne*((Ie_gain/Ie_syn)+1)) : amp (clock-driven)
        dIi_syn/dt = (-Ii_syn - Ii_gain + 2*Io_syn*(Ii_syn<=Io_syn))/(tausyni*((Ii_gain/Ii_syn)+1)) : amp (clock-driven)

        Ie{input_number}_post = Ie_syn : amp (summed)
        Ii{input_number}_post = -Ii_syn : amp (summed)

        weight : 1
        w_plast : 1

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
        Ie_syn += Iw_e*w_plast*Ie_gain*(weight>0)/(Itau_e*((Ie_gain/Ie_syn)+1))
        Ii_syn += Iw_i*w_plast*Ii_gain*(weight<0)/(Itau_i*((Ii_gain/Ii_syn)+1))
        ''',
       'on_post': ''' ''',
       }

# standard parameters for DPI models
dpi_params = {
    'Io_syn': 0.5 * pA,
    'kn_syn': 0.75,
    'kp_syn': 0.66,
    'Ut_syn': 25. * mV,
    'Csyn': 1.5 * pF,
    'Ie_tau': 10. * pA,
    'Ii_tau': 10. * pA,
    'Ie_th': 10 * pA,
    'Ii_th': 10 * pA,
    'Ie_syn': 0.5 * pA,
    'Ii_syn': 0.5 * pA,
    'w_plast': 1,
    'baseweight_e': 50. * pA,
    'baseweight_i': 50. * pA
}

# DPI shunting inhibition
dpi_shunt = {'model': """
            dIi_syn/dt = (-Ii_syn - Ii_gain + 2*Io_syn*(Ii_syn<=Io_syn))/(tausyni*((Ii_gain/Ii_syn)+1)) : amp (clock-driven)

            Ishunt{input_number}_post = -Ii_syn : amp  (summed)

            weight : 1
            w_plast : 1

            Ii_gain = Io_syn*(Ii_syn<=Io_syn) + Ii_th*(Ii_syn>Io_syn) : amp

            Itau_i = Io_syn*(Ii_syn<=Io_syn) + Ii_tau*(Ii_syn>Io_syn) : amp

            baseweight_i : amp (constant)     # synaptic gain
            tausyni = Csyn * Ut_syn /(kappa_syn * Itau_i) : second
            kappa_syn = (kn_syn + kp_syn) / 2 : 1


            Iw_i = weight*baseweight_i  : amp

            Ii_tau       : amp (constant)
            Ii_th        : amp (constant)
            kn_syn       : 1 (constant)
            kp_syn       : 1 (constant)
            Ut_syn       : volt (constant)
            Io_syn       : amp (constant)
            Csyn         : farad (constant)
            """,
             'on_pre': """
             Ii_syn += Iw_i*w_plast*Ii_gain*(weight<0)/(Itau_i*((Ii_gain/Ii_syn)+1))
              """,
             'on_post': """ """
             }

dpi_shunt_params = {
    'Csyn': 1.5 * pF,
    'Io_syn': 0.5 * pA,
    'Ii_tau': 10. * pA,
    'Ut_syn': 25. * mV,
    'baseweight_i': 50. * pA,
    'kn_syn': 0.75,
    'kp_syn': 0.66,
    'wPlast': 1,
    'Ii_th': 10 * pA,
    'Ii_syn': 0.5 * pA
}


"""ADDITIONAL EQUATIONS BLOCKS AND PARAMETERS

Every block must specify additional model, pre- and post-spike equations, as well as
two different sets (dictionaries) of parameters for conductance based
models or current models.

If you want to override an equation add '%' before the variable of your
block's explicit equation.

Example:  Let's say we have the simplest model (current one with template equation),
and you're implementing a new block with this explicit equation : d{synvar_e}/dt = (-{synvar_e})**2 / synvar_e.
If you want to override the equation already declared in the template: d{synvar_e}/dt = (-{synvar_e}) / tausyne + kernel_e,
your equation will be : %d{synvar_e}/dt = (-{synvar_e})**2 / synvar_e


Plasticity Blocks:

You need to declare two set of parameters for every block: (one for
current based models and one for conductance based models)
"""
# Fusi learning rule ##
fusi = {'model': '''
      dCa/dt = (-Ca/tau_ca) : volt (event-driven) #Calcium Potential

      updrift = 1.0*(w>theta_w) : 1
      downdrift = 1.0*(w<=theta_w) : 1

      dw/dt = (alpha*updrift)-(beta*downdrift) : 1 (event-driven) # internal weight variable

      wplus: 1 
      wminus: 1 
      theta_upl: volt (constant)
      theta_uph: volt (constant)
      theta_downh: volt (constant)
      theta_downl: volt (constant)
      theta_V: volt (constant)
      alpha: 1/second (constant)
      beta: 1/second (constant)
      tau_ca: second (constant)
      w_min: 1 (constant)
      w_max: 1 (constant)
      theta_w: 1 (constant)
      w_ca: volt (constant)     ''',

        'on_pre': '''
      up = 1. * (Vm_post>theta_V) * (Ca>theta_upl) * (Ca<theta_uph)
      down = 1. * (Vm_post<theta_V) * (Ca>theta_downl) * (Ca<theta_downh)
      w += wplus * up - wminus * down
      w = clip(w,w_min,w_max)
      w_plast = floor(w+0.5)
      ''',

        'on_post': '''Ca += w_ca'''}

fusi_params_current = {"wplus": 0.2,
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

fusi_params_conductance = {"wplus": 0.2,
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
      w_max: 1 (constant)
      taupre : second (constant)
      taupost : second (constant)
      dApre : 1 (constant)
      Q_diffAPrePost : 1 (constant)
      ''',

        'on_pre': '''
      Apre += dApre*w_max
      w_plast = clip(w_plast + Apost, 0, w_max) ''',

        'on_post': '''
      Apost += -dApre * (taupre / taupost) * Q_diffAPrePost * w_max
      w_plast = clip(w_plast + Apre, 0, w_max) '''}

stdp_para_current = {"baseweight_e": 7 * pA,  # should we find a way to replace since we would define it twice?
                     "baseweight_i": 7 * pA,
                     "taupre": 10 * ms,
                     "taupost": 10 * ms,
                     "w_max": 1.,
                     "dApre": 0.1,
                     "Q_diffAPrePost": 1.05,
                     "w_plast": 0}

stdp_para_conductance = {"baseweight_e": 7 * nS,  # should we find a way to replace since we would define it twice?
                         "baseweight_i": 3 * nS,
                         "taupre": 20 * ms,
                         "taupost": 20 * ms,
                         "w_max": 0.01,
                         "diffApre": 0.01,
                         "Q_diffAPrePost": 1.05,
                         "w_plast": 0}
"""Kernels Blocks:

You need to declare two set of parameters for every block: one for
current based models and one for conductance based models.

TODO: THESE KERNELS ARE WRONG!
"""
# Alpha kernel ##

alpha_kernel = {'model': '''
             %kernel_e = baseweight_e*(weight>0)*w_plast*weight*exp(1-t_spike/tausyne_rise)/tausyne : {unit} * second **-1
             %kernel_i = baseweight_i*(weight<0)*w_plast*weight*exp(1-t_spike/tausyni_rise)/tausyni : {unit} * second **-1
             dt_spike/dt = 1 : second (clock-driven)
             tausyne_rise : second
             tausyni_rise : second
             ''',

                'on_pre': '''
             t_spike = 0 * ms
             %Ie_syn += 0 * amp
             %Ii_syn += 0 * amp
             ''',

                'on_post': ''' '''}
             # factor_e : 1
             # tpeak_e = (tausyne * tausyne_rise) / (tausyne - tausyne_rise) * log(tausyne / tausyne_rise) : second
             # factor_e = 1 / (-exp(-tpeak/tausyne_rise) + exp(-tpeak/tausyne))
             # factor_i : 1
             # tpeak_i = (tausyni * tausyni_rise) / (tausyni - tausyni_rise) * log(tausyni / tausyni_rise) : second
             # factor_i = 1 / (-exp(-tpeak/tausyni_rise) + exp(-tpeak/tausyni))

dexp_kernel = {'model': '''
             %dkernel_e/dt = -kernel_e/tausyne_rise + baseweight_e*(weight>0)*w_plast*h/(tausyne_rise*tausyne) : {unit} * second **-1 (clock-driven)
             %dkernel_i/dt = -kernel_i/tausyni_rise + baseweight_i*(weight<0)*w_plast*h/(tausyni_rise*tausyni) : {unit} * second **-1 (clock-driven)
             h : 1
             tausyne_rise : second
             tausyni_rise : second
             ''',

                'on_pre': '''
             h += weight
             %Ie_syn += 0 * amp
             %Ii_syn += 0 * amp
             ''',

                'on_post': ''' '''}

alpha_params_current = {"tausyne": 2 * ms,
                        "tausyni": 2 * ms,
                        "tausyne_rise": 0.5 * ms,
                        "tausyni_rise": 0.5 * ms,
                        "t_spike": 5000 * ms}  # Assuming that last spike has occurred long time ago

dexp_params_current = {"tausyne": 2 * ms,
                        "tausyni": 2 * ms,
                        "tausyne_rise": 0.5 * ms,
                        "tausyni_rise": 0.5 * ms}  # Assuming that last spike has occurred long time ago

alpha_params_conductance = {"tausyne": 2 * ms,
                            "tausyni": 2 * ms,
                            "tausyne_rise": 1 * ms,
                            "tausyni_rise": 1 * ms}

# Resonant kernel ##
resonant_kernel = {'model': '''
                omega: 1/second
                sigma_gaussian : second
                %kernel_e  = baseweight_e*(weight>0)*w_plast*(weight*exp(-t_spike/tausyne_rise)*cos(omega*t_spike))/tausyne : {unit} * second **-1
                %kernel_i  = baseweight_i*(weight<0)*w_plast*(weight*exp(-t_spike/tausyni_rise)*cos(omega*t_spike))/tausyni : {unit} * second **-1
                dt_spike/dt = 1 : second (clock-driven)
                tausyne_rise : second
                tausyni_rise : second
                ''',

                   'on_pre': '''

                t_spike = 0 * ms
                ''',

                   'on_post': ''' '''}

resonant_params_current = {"tausyne": 2 * ms,
                           "tausyni": 2 * ms,
                           "omega": 7 / ms,
                           "tausyne_rise": 0.5 * ms,
                           "tausyni_rise": 0.5 * ms}

resonant_params_conductance = {"tausyne": 2 * ms,
                               "tausyni": 2 * ms,
                               "omega": 1 / ms}


#  Gaussian kernel ##


gaussian_kernel = {'model': '''
                  %tausyne = (sigma_gaussian_e**2)/t_spike : second
                  %tausyni = (sigma_gaussian_i**2)/t_spike : second
                  sigma_gaussian_e : second
                  sigma_gaussian_i : second

                  dt_spike/dt = 1 : second (clock-driven)
                  ''',
                   # this time we need to add this pre eq to the template pe eq

                   'on_pre': '''t_spike = 0 * ms''',

                   'on_post': ''' '''}

gaussian_params_current = {"sigma_gaussian_e": 6 * ms,
                           "sigma_gaussian_i": 6 * ms}

gaussian_params_conductance = {"sigma_gaussian_e": 6 * ms,
                               "sigma_gaussian_i": 6 * ms}


none_params = {}

"""Dictionary of keywords:

These dictionaries contains keyword and models and parameters names useful for the __init__ subroutine
Every new block dictionaries must be added to these definitions
"""
modes = {'current': current, 'conductance': conductance,
         'DPI': dpi, 'DPIShunting': dpi_shunt}

kernels = {'exponential': none, 'alpha': alpha_kernel, 'dexp': dexp_kernel,
           'resonant': resonant_kernel, 'gaussian': gaussian_kernel}

plasticity_models = {'non_plastic': none, 'fusi': fusi, 'stdp': stdp}


# parameters dictionaries
current_parameters = {'current': current_params, 'non_plastic': none_params, 'fusi': fusi_params_current,
                      'stdp': stdp_para_current, 'exponential': none_params, 'alpha': alpha_params_current, 'dexp': dexp_params_current,
                      'resonant': resonant_params_current, 'gaussian': gaussian_params_current}

conductance_parameters = {'conductance': conductance_params, 'non_plastic': none_params, 'fusi': fusi_params_conductance,
                          'stdp': stdp_para_conductance, 'exponential': none_params, 'alpha': alpha_params_conductance,
                          'resonant': resonant_params_conductance, 'gaussian': gaussian_params_conductance}

DPI_parameters = {'DPI': dpi_params, 'exponential': none_params, 'non_plastic': none_params,
                  'fusi': fusi_params_current, 'stdp': stdp_para_current}

DPI_shunt_parameters = {'DPIShunting': dpi_shunt_params, 'exponential': none_params, 'non_plastic': none_params,
                        'fusi': fusi_params_current, 'stdp': stdp_para_current}
