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


from teili import constants
from brian2 import pF, nS, mV, ms, pA, nA, volt, second

none = {'model': ''' ''', 'on_pre': ''' ''', 'on_post': ''' '''}

current = {'model': '''
            dI_syn/dt = (-I_syn) / tausyn + kernel: amp (clock-driven)

            kernel : amp * second **-1

            Iin{input_number}_post = I_syn *  sign(weight) : amp (summed)

            tausyn : second (constant) # synapse time constant
            w_plast : 1

            baseweight : amp (constant)     # synaptic gain
            weight : 1

            ''',

           'on_pre': '''
            I_syn += baseweight * abs(weight) * w_plast
            ''',

           'on_post': ''' '''
           }

# standard parameters for current based models
current_params = {"tausyn": 5 * ms,
                  "w_plast": 1,
                  "baseweight": 1 * nA,
                  "kernel": 0 * nA * ms**-1
                  }

# Additional equations for conductance based models
conductance = {'model': '''
               dgI/dt = (-gI) / tausyn + kernel: siemens (clock-driven)

               I_syn = gI*(EI - Vm_post) : amp

               Iin{input_number}_post = I_syn *  sign(weight)  : amp (summed)

               EI =  EIe: volt (constant)             # reversal potential

               kernel : siemens * second **-1

               tausyn : second (constant) # synapse time constant
               w_plast : 1

               baseweight : siemens (constant)     # synaptic gain
               weight : 1

               ''',

               'on_pre': '''
               gI += baseweight * abs(weight) * w_plast
               ''',

               'on_post': ''' '''
               }

# standard parameters for conductance based models
conductance_params = {"gI": 0 * nS,
                      "tausyn": 5 * ms,
                      "EIe": 60.0 * mV, # For inhibitory this parameter is negative...will thus
                      #be a problem?
                      "EIi": -90.0 * mV,
                      "w_plast": 1,
                      "baseweight": 7 * nS,
                      "kernel": 0 * nS * ms**-1
                      }

# DPI type model
dpi = {'model': '''
        dI_syn/dt = (-I_syn - I_gain + 2*Io_syn*(I_syn<=Io_syn))/(tausyn*((I_gain/I_syn)+1)) : amp (clock-driven)

        Iin{input_number}_post = I_syn *  sign(weight)  : amp (summed)

        weight : 1
        w_plast : 1

        I_gain = Io_syn*(I_syn<=Io_syn) + I_th*(I_syn>Io_syn) : amp
        Itau_syn = Io_syn*(I_syn<=Io_syn) + I_tau*(I_syn>Io_syn) : amp

        baseweight : amp (constant)     # synaptic gain
        tausyn = Csyn * Ut_syn /(kappa_syn * Itau_syn) : second
        kappa_syn = (kn_syn + kp_syn) / 2 : 1

        Iw = abs(weight) * baseweight  : amp

        I_tau       : amp (constant)
        I_th        : amp (constant)
        kn_syn       : 1 (constant)
        kp_syn       : 1 (constant)
        Ut_syn       : volt (constant)
        Io_syn       : amp (constant)
        Csyn         : farad (constant)
        ''',
       'on_pre': '''
        I_syn += Iw * w_plast * I_gain / (Itau_syn * ((I_gain/I_syn)+1))
        ''',
       'on_post': ''' ''',
       }

# standard parameters for DPI models
dpi_params = {
    'Io_syn': constants.I0,
    'kn_syn': constants.KAPPA_N,
    'kp_syn': constants.KAPPA_P,
    'Ut_syn': 25. * mV,
    'Csyn': 1.5 * pF,
    'I_tau': 10. * pA,
    'I_th': 10 * pA,
    'I_syn': constants.I0,
    'w_plast': 1,
    'baseweight': 7. * pA #50. * pA
}

# DPI shunting inhibition
dpi_shunt = {'model': """
            dI_syn/dt = (-I_syn - I_gain + 2*Io_syn*(I_syn<=Io_syn))/(tausyn*((I_gain/I_syn)+1)) : amp (clock-driven)

            Ishunt{input_number}_post = I_syn *  sign(weight)  * (weight<0) : amp (summed)

            weight : 1
            w_plast : 1

            I_gain = Io_syn*(I_syn<=Io_syn) + I_th*(I_syn>Io_syn) : amp

            Itau_syn = Io_syn*(I_syn<=Io_syn) + I_tau*(I_syn>Io_syn) : amp

            baseweight : amp (constant)     # synaptic gain
            tausyn = Csyn * Ut_syn /(kappa_syn * Itau_syn) : second
            kappa_syn = (kn_syn + kp_syn) / 2 : 1


            Iw = abs(weight) * baseweight  : amp

            I_tau       : amp (constant)
            I_th        : amp (constant)
            kn_syn       : 1 (constant)
            kp_syn       : 1 (constant)
            Ut_syn       : volt (constant)
            Io_syn       : amp (constant)
            Csyn         : farad (constant)
            """,
             'on_pre': """
             I_syn += Iw * w_plast * I_gain * (weight<0)/(Itau_syn*((I_gain/I_syn)+1              """,
             'on_post': """ """
             }

dpi_shunt_params = {
    'Csyn': 1.5 * pF,
    'Io_syn': constants.I0,
    'I_tau': 10. * pA,
    'Ut_syn': 25. * mV,
    'baseweight': 50. * pA,
    'kn_syn': 0.75,
    'kp_syn': 0.66,
    'wPlast': 1,
    'I_th': 10 * pA,
    'I_syn': constants.I0
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

stdp_para_current = {"baseweight": 7 * pA,  # should we find a way to replace since we would define it twice?
                     "taupre": 10 * ms,
                     "taupost": 10 * ms,
                     "w_max": 1.,
                     "dApre": 0.1,
                     "Q_diffAPrePost": 1.05,
                     "w_plast": 0}

stdp_para_conductance = {"baseweight": 7 * nS,  # should we find a way to replace since we would define it twice?
                         "taupre": 20 * ms,
                         "taupost": 20 * ms,
                         "w_max": 0.01,
                         "diffApre": 0.01,
                         "Q_diffAPrePost": 1.05,
                         "w_plast": 0}
"""Kernels Blocks:

You need to declare two set of parameters for every block: one for
current based models and one for conductance based models.

"""
# Alpha kernel ##

alpha_kernel = {'model': '''
             %kernel = s/tausyn  : {unit} * second **-1
             ds/dt = -s/tausyn : amp
             tausyn_rise : second
             ''',

                'on_pre': '''
             s += baseweight * w_plast * abs(weight)
             %I_syn += 0 * amp
             ''',

                'on_post': ''' '''}

alpha_params_current = {"tausyn": 0.5 * ms,
                        "tausyn_rise": 2 * ms} 

alpha_params_conductance = {"tausyn": 0.5 * ms,
                            "tausyn_rise": 1 * ms}


# Who wrote dexp?
dexp_kernel = {'model': '''
             %dkernel/dt = -kernel/tausyn_rise + baseweight*w_plast*h/(tausyn_rise*tausyn) : {unit} * second **-1 (clock-driven)
             h : 1
             tausyn_rise : second
             ''',

                'on_pre': '''
             h += abs(weight)
             %I_syn += 0 * amp
             ''',

                'on_post': ''' '''}


dexp_params_current = {"tausyn": 2 * ms,
                        "tausyn_rise": 0.5 * ms}

dexp_params_conductance = {"tausyn": 2 * ms,
                            "tausyn_rise": 1 * ms}

# Resonant kernel ##
resonant_kernel = {'model': '''
                omega: 1/second
                %kernel  = s * omega : {unit} * second **-1
                ds/dt = -s/tausyn - I_syn*omega : amp
                tausyn_kernel : second
                ''',

                   'on_pre': '''
                s += baseweight * w_plast * abs(weight)
                %I_syn += 0 * amp
                ''',

                   'on_post': ''' '''}

resonant_params_current = {"tausyn": 0.5 * ms,
                           "omega": 3 / ms,
                           "tausyn_kernel": 0.5 * ms}

resonant_params_conductance = {"tausyn": 0.5 * ms,
                               "omega": 1 / ms,
                               "tausyn_kernel": 0.5 * ms}


none_params = {}

"""Dictionary of keywords:

These dictionaries contains keyword and models and parameters names useful for the __init__ subroutine
Every new block dictionaries must be added to these definitions
"""
modes = {'current': current, 'conductance': conductance,
         'DPI': dpi, 'DPIShunting': dpi_shunt}

kernels = {'exponential': none, 'alpha': alpha_kernel,
           'resonant': resonant_kernel}

plasticity_models = {'non_plastic': none, 'fusi': fusi, 'stdp': stdp}


# parameters dictionaries
current_parameters = {'current': current_params, 'non_plastic': none_params, 'fusi': fusi_params_current,
                      'stdp': stdp_para_current, 'exponential': none_params, 'alpha': alpha_params_current,
                      'resonant': resonant_params_current}

conductance_parameters = {'conductance': conductance_params, 'non_plastic': none_params, 'fusi': fusi_params_conductance,
                          'stdp': stdp_para_conductance, 'exponential': none_params, 'alpha': alpha_params_conductance,
                          'resonant': resonant_params_conductance}

DPI_parameters = {'DPI': dpi_params, 'exponential': none_params, 'alpha': alpha_params_current, 'non_plastic': none_params,
                  'fusi': fusi_params_current, 'stdp': stdp_para_current}

DPI_shunt_parameters = {'DPIShunting': dpi_shunt_params, 'exponential': none_params, 'non_plastic': none_params,
                        'fusi': fusi_params_current, 'stdp': stdp_para_current}
