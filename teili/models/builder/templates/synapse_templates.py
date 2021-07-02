# -*- coding: utf-8 -*-
"""This file contains dictionaries of synapse equations or modules,
combined by the synapse equation builder.
Each template consists of a dictionary containing the relevant
equations and a corresponding parameter dictionary.

For usage example plaese refer to `teili/models/synapse_models.py`

Contributing guide:
*  Dictionary describing the synapse model
    *  Describing the model dynamics, including all used variables
       and their units.
    *  Required keys in the dictionary: 'model', 'on_pre', 'on_post'.
    *  name: modelname_template
*  Corresponding dictionary containing default/init parameters.
    *  name: modelname_template_params

If you want to override an equation add '%' before the variable of your
block's explicit equation.

Example:  Let's say we have the simplest model (current one with template equation),
and you're implementing a new block with this explicit equation : d{synvar_e}/dt = (-{synvar_e})**2 / synvar_e.
If you want to override the equation already declared in the template: d{synvar_e}/dt = (-{synvar_e}) / tausyne + kernel_e,
your equation will be : %d{synvar_e}/dt = (-{synvar_e})**2 / synvar_e

"""
# @Author: Moritz Milde
# @Date:   2018-06-01 11:57:02


from teili import constants
from brian2 import pF, nS, mV, ms, pA, nA, mA, volt, second

current = {
    'model': '''
        dI_syn/dt = (-I_syn) / tausyn + kernel: amp (clock-driven)
        Iin{input_number}_post = I_syn *  sign(weight) : amp (summed)

        kernel : amp * second **-1
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
current_params = {
    "tausyn": 5 * ms,
    "w_plast": 1,
    "baseweight": 1 * nA,
    "kernel": 0 * nA * ms ** -1
}

# Additional equations for conductance based models
conductance = {
    'model': '''
        dgI/dt = (-gI) / tausyn + kernel     : siemens (clock-driven)
        I_syn = gI*(EI - Vm_post)            : amp
        Iin{input_number}_post = I_syn *  sign(weight)  : amp (summed)

        EI =  EIe                            : volt        # reversal potential
        kernel                               : siemens * second **-1
        tausyn                               : second   (constant) # synapse time constant
        w_plast                              : 1
        baseweight                           : siemens (constant)     # synaptic gain
        weight                               : 1
        EIe                                  : volt
        EIi                                  : volt
        ''',
    'on_pre': '''
        gI += baseweight * abs(weight) * w_plast
        ''',
    'on_post': ''' '''
}
""" Standard parameters for conductance based models
TODO: For inhibitory synapse EIe is negative. Could this thus be a problem?
"""
conductance_params = {
    "gI": 0 * nS,
    "tausyn": 5 * ms,
    "EIe": 60.0 * mV,
    "EIi": -90.0 * mV,
    "w_plast": 1,
    "baseweight": 7 * nS,
    "weight": 1,
    "kernel": 0 * nS * ms ** -1
}

# DPI type model
dpi = {
    'model': """
        dI_syn/dt = (-I_syn - I_gain + 2*Io_syn*(I_syn<=Io_syn))/(tausyn*((I_gain/I_syn)+1)) : amp (clock-driven)
        Iin{input_number}_post = I_syn *  sign(weight)           : amp (summed)
        Iw = abs(weight) * baseweight                            : amp
        I_gain = Io_syn*(I_syn<=Io_syn) + I_th*(I_syn>Io_syn)    : amp
        Itau_syn = Io_syn*(I_syn<=Io_syn) + I_tau*(I_syn>Io_syn) : amp
        tausyn = Csyn * Ut_syn /(kappa_syn * Itau_syn)           : second
        kappa_syn = (kn_syn + kp_syn) / 2                        : 1

        weight     : 1
        w_plast    : 1
        baseweight : amp   (constant)
        I_tau      : amp   (constant)
        I_th       : amp   (constant)
        kn_syn     : 1     (constant)
        kp_syn     : 1     (constant)
        Ut_syn     : volt  (constant)
        Io_syn     : amp   (constant)
        Csyn       : farad (constant)
        """,
    'on_pre': """
        I_syn += Iw * w_plast * I_gain / (Itau_syn * ((I_gain/I_syn)+1))
        """,
    'on_post': """ """,
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
    'baseweight': 7. * pA,  # 50. * pA
    "weight": 1
}

# DPI shunting inhibition
dpi_shunt = {
    'model': """
        dI_syn/dt = (-I_syn - I_gain + 2*Io_syn*(I_syn<=Io_syn))/(tausyn*((I_gain/I_syn)+1)) : amp (clock-driven)
        Ishunt{input_number}_post = I_syn *  sign(weight)  * (weight<0) : amp (summed)
        Iw = abs(weight) * baseweight                                   : amp
        I_gain = Io_syn*(I_syn<=Io_syn) + I_th*(I_syn>Io_syn)           : amp
        Itau_syn = Io_syn*(I_syn<=Io_syn) + I_tau*(I_syn>Io_syn)        : amp
        tausyn = Csyn * Ut_syn /(kappa_syn * Itau_syn)                  : second
        kappa_syn = (kn_syn + kp_syn) / 2                               : 1

        weight     : 1
        w_plast    : 1
        baseweight : amp   (constant)
        I_tau      : amp   (constant)
        I_th       : amp   (constant)
        kn_syn     : 1     (constant)
        kp_syn     : 1     (constant)
        Ut_syn     : volt  (constant)
        Io_syn     : amp   (constant)
        Csyn       : farad (constant)
        """,
    'on_pre': """
        I_syn += Iw * w_plast * I_gain * (weight<0)/(Itau_syn*((I_gain/I_syn)+1
        """,
    'on_post': """ """,
}

dpi_shunt_params = {
    'Csyn': 1.5 * pF,
    'Io_syn': constants.I0,
    'I_tau': 10. * pA,
    'Ut_syn': 25. * mV,
    'baseweight': 50. * pA,
    "weight": 1,
    'kn_syn': 0.75,
    'kp_syn': 0.66,
    'I_th': 10 * pA,
    'I_syn': constants.I0
}

"""Exponentially decaying synapse model using quantized stochastic
decay taken from Wang et al. (2018). Please refer to this paper
for more information. Note that this model was conceptualized in
discrete time with backward euler scheme and an integer
operation. An state updader with x_new = f(x,t) and
defaultclock.dt = 1*ms in the code using this model is necessary.
"""
quantized_stochastic = {
    'model': '''
        dI_syn/dt = int(I_syn*decay_syn/mA + decay_probability_syn)*mA/second : amp (clock-driven)
        decay_probability_syn = rand() : 1 (constant over dt)
        Iin{input_number}_post = I_syn * sign(weight)                           : amp (summed)
        syn_release : 1

        decay_syn = tausyn/(tausyn + dt) : 1

        weight                : 1
        w_plast               : 1
        gain_syn              : amp
        tausyn               : second (constant)
        ''',
        'on_pre': '''
        syn_release = int(rand()<0.5)
        I_syn += gain_syn * abs(weight) * w_plast * syn_release
        I_syn = clip(I_syn, 0*mA, 15*mA)
        ''',
        'on_post': '''
        '''
}

quantized_stochastic_params = {
    'weight' : 1,
    'w_plast' : 1,
    'gain_syn' : 1*mA,
    'tausyn': 3*ms
}

""" **Plasticity blocks**
You need to declare two set of parameters for every block:
*   current based models
*   conductance based models
"""

"""Fusi learning rule based on Ca dynamics as published in
Mitra et al., 2008
"""
fusi = {
    'model': """
        dCa/dt = (-Ca/tau_ca)                    : volt (event-driven)

        updrift = 1.0*(w>theta_w)                : 1
        downdrift = 1.0*(w<=theta_w)             : 1
        dw/dt = (alpha*updrift)-(beta*downdrift) : 1 (event-driven)

        wplus       : 1
        wminus      : 1
        theta_upl   : volt     (constant)
        theta_uph   : volt     (constant)
        theta_downh : volt     (constant)
        theta_downl : volt     (constant)
        theta_V     : volt     (constant)
        alpha       : 1/second (constant)
        beta        : 1/second (constant)
        tau_ca      : second   (constant)
        w_min       : 1        (constant)
        w_max       : 1        (constant)
        theta_w     : 1        (constant)
        w_ca        : volt     (constant)
        """,
    'on_pre': """
        up = 1. * (Vm_post>theta_V) * (Ca>theta_upl) * (Ca<theta_uph)
        down = 1. * (Vm_post<theta_V) * (Ca>theta_downl) * (Ca<theta_downh)
        w += wplus * up - wminus * down
        w = clip(w,w_min,w_max)
        w_plast = floor(w+0.5)
        """,
    'on_post': """
        Ca += w_ca
        """,
}

fusi_params_current = {
    "wplus": 0.2,
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

fusi_params_conductance = {
    "wplus": 0.2,
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

stdgm = {
    'model': '''
        dApre/dt = -Apre / taupre : 1 (event-driven)
        dApost/dt = -Apost / taupost : 1 (event-driven)
        gain_max: 1 (shared, constant)
        taupre : second (shared, constant)
        taupost : second (shared, constant)
        dApre : 1 (shared, constant)
        dApost : 1 (shared, constant)
        Ipred_plast : 1
        Q_diffAPrePost : 1 (shared, constant)
        scaling_factor : 1 (shared, constant)
        ''',
    'on_pre': '''
        Apre += dApre*gain_max
        Ipred_plast = clip(Ipred_plast + Apost, 0, gain_max)
        Ipred_post = (Ipred_post - (scaling_factor * Ipred_plast)) * (Ipred_post>0)
        ''',
    'on_post':
        '''
        Apost += -dApre * (taupre / taupost) * Q_diffAPrePost * gain_max
        Ipred_plast = clip(Ipred_plast + Apre, 0, gain_max)
        '''
}

stdgm_params = {
    'dApre': '0.01',
    'Ipred_plast': '0.0',
    'gain_max': '1.0',
    'taupre': '5 * msecond',
    'taupost': '5 * msecond',
    'Q_diffAPrePost': '1.05',
    'scaling_factor': '0.1'
}

"""
The set of activation encapsulates `StateVariables` which are needed for
Activity Dependent Plasticity (ADP) paradigm. ADP adjusts the inhibitory weight
according to the 'activity' of the post-synaptic neuron.
These equations are required by synapses projecting to _adp neuronal populations.
"""

activity = {
    'model': '''
        inh_learning_rate: 1 (constant, shared)
        variance_th: 1 (constant)
        delta_w : 1
        ''',
    'on_pre': '''
        delta_w = inh_learning_rate * (normalized_activity_proxy_post - variance_th)
        w_plast = clip(w_plast + delta_w, 0, 1.0)
        ''',
    'on_post': ''' '''
}

activity_params = {
    'inh_learning_rate': '0.01',
    'variance_th': '0.67',
}

# Use lfsr to generate random numbers (only works when dt=1*ms). Function must
# be added to namespace via teili.tools.lfsr
lfsr_syn = {
    'model': """
        %decay_probability_syn = lfsr_timedarray( ((seed_syn+t) % lfsr_max_value_syn) + lfsr_init_syn ) / (2**lfsr_num_bits_syn) : 1
        %decay_probability_Apost = lfsr_timedarray( ((seed_Apost+t) % lfsr_max_value_Apost) + lfsr_init_Apost ) / (2**lfsr_num_bits_Apost) : 1
        %decay_probability_Apre = lfsr_timedarray( ((seed_Apre+t) % lfsr_max_value_Apre) + lfsr_init_Apre ) / (2**lfsr_num_bits_Apre) : 1

        seed_Apre : second
        lfsr_max_value_Apre : second
        lfsr_init_Apre : second
        lfsr_num_bits_Apre : 1 # Number of bits in the LFSR used

        seed_Apost : second
        lfsr_max_value_Apost : second
        lfsr_init_Apost : second
        lfsr_num_bits_Apost : 1 # Number of bits in the LFSR used

        seed_syn : second
        lfsr_max_value_syn : second
        lfsr_init_syn : second
        lfsr_num_bits_syn : 1 # Number of bits in the LFSR used

        counter_Apre : second
        counter_Apost : second

        seed_condApre1 : second
        seed_condApre2 : second
        seed_condApost1 : second
        seed_condApost2 : second
        lfsr_max_value_condApre1 : second
        lfsr_max_value_condApre2 : second
        lfsr_max_value_condApost1 : second
        lfsr_max_value_condApost2 : second
        lfsr_init_condApre1 : second
        lfsr_init_condApre2 : second
        lfsr_init_condApost1 : second
        lfsr_init_condApost2 : second

        lfsr_num_bits_condApre1 : 1 # Variables used for initialization only
        lfsr_num_bits_condApre2 : 1
        lfsr_num_bits_condApost1 : 1
        lfsr_num_bits_condApost2 : 1
         """,
    'on_pre': """
        counter_Apre += dt
        %rand_int_Apre1 = lfsr_timedarray( ((seed_condApre1+counter_Apre) % lfsr_max_value_condApre1) + lfsr_init_condApre1 )
        %rand_int_Apre2 = lfsr_timedarray( ((seed_condApre2+counter_Apre) % lfsr_max_value_condApre2) + lfsr_init_condApre2 )
         """,
    'on_post': """
        counter_Apost += dt
        %rand_int_Apost1 = lfsr_timedarray( ((seed_condApost1+counter_Apost) % lfsr_max_value_condApost1) + lfsr_init_condApost1 )
        %rand_int_Apost2 = lfsr_timedarray( ((seed_condApost2+counter_Apost) % lfsr_max_value_condApost2) + lfsr_init_condApost2 )
         """
}

lfsr_syn_params = {
    'counter_Apre': '0 * msecond',
    'counter_Apost': '0 * msecond',
    'lfsr_num_bits_Apre' : '6',
    'lfsr_num_bits_Apost' : '6',
    'lfsr_num_bits_syn' : '6',
    'lfsr_num_bits_condApre1': '4',
    'lfsr_num_bits_condApre2': '4',
    'lfsr_num_bits_condApost1': '4',
    'lfsr_num_bits_condApost2': '4'
    }

# STDP learning rule ##
stdp = {
    'model': '''
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
        w_plast = clip(w_plast + Apost, 0, w_max)
        ''',
    'on_post': '''
        Apost += -dApre * (taupre / taupost) * Q_diffAPrePost * w_max
        w_plast = clip(w_plast + Apre, 0, w_max)
        '''
}

stdp_para_current = {
    "taupre": 10 * ms,
    "taupost": 10 * ms,
    "w_max": 1.,
    "dApre": 0.1,
    "Q_diffAPrePost": 1.05,
    "w_plast": 0
}

stdp_para_conductance = {
    "taupre": 20 * ms,
    "taupost": 20 * ms,
    "w_max": 0.01,
    "diffApre": 0.01,
    "Q_diffAPrePost": 1.05,
    "w_plast": 0
}

quantized_stochastic_stdp = {
    'model': '''
        dApre/dt = int(Apre * decay_stdp_Apre + decay_probability_Apre)/second : 1 (clock-driven)
        dApost/dt = int(Apost * decay_stdp_Apost + decay_probability_Apost)/second : 1 (clock-driven)
        decay_probability_Apre = rand() : 1 (constant over dt)
        decay_probability_Apost = rand() : 1 (constant over dt)

        decay_stdp_Apre = taupre/(taupre + dt) : 1
        decay_stdp_Apost = taupost/(taupost + dt) : 1

        w_max: 1 (constant)
        A_max: 1 (constant)
        dApre: 1 (constant)
        A_gain: 1 (constant)
        taupre : second (constant)
        taupost : second (constant)

        rand_int_Apre1 : 1
        rand_int_Apre2 : 1
        rand_int_Apost1 : 1
        rand_int_Apost2 : 1
        rand_num_bits_Apre : 1 # Number of bits of random number generated for Apre
        rand_num_bits_Apost : 1 # Number of bits of random number generated for Apost
        stdp_thres : 1 (constant)
        ''',
    'on_pre': '''
        Apre += dApre * syn_release
        Apre = clip(Apre, 0, A_max)
        rand_int_Apre1 = ceil(rand() * (2**rand_num_bits_Apre-1))
        rand_int_Apre2 = ceil(rand() * (2**rand_num_bits_Apre-1))
        w_plast = clip(w_plast - 1*int(lastspike_post!=lastspike_pre)*int(rand_int_Apre1 < Apost)*int(rand_int_Apre2 <= stdp_thres)*syn_release, 0, w_max)
        ''',
    'on_post': '''
        Apost += dApre
        Apost = clip(Apost, 0, A_max)
        rand_int_Apost1 = ceil(rand() * (2**rand_num_bits_Apost-1))
        rand_int_Apost2 = ceil(rand() * (2**rand_num_bits_Apost-1))
        w_plast = clip(w_plast + 1*int(lastspike_post!=lastspike_pre)*int(rand_int_Apost1 < Apre)*int(rand_int_Apost2 <= stdp_thres), 0, w_max)
        '''
}

quantized_stochastic_stdp_params = {
    "taupre": 3 * ms,
    "taupost": 3 * ms,
    "w_max": 15,
    "A_max": 15,
    "A_gain": 4,
    "dApre": 15,
    "w_plast": 1,
    "rand_num_bits_Apre": 6,
    "rand_num_bits_Apost": 6,
    "stdp_thres": 2
}

"""Kernels Blocks:
You need to declare two set of parameters for every block:
*   current based models
*   conductance based models
"""
alpha_kernel = {
    'model': """
        %kernel = s/tausyn  : {unit} * second **-1
        ds/dt = -s/tausyn   : amp (clock-driven)

        tausyn_rise : second
        """,
    'on_pre': """
        s += baseweight * w_plast * abs(weight)
        %I_syn += 0 * amp
        """,
    'on_post': """ """,
}

alpha_params_current = {
    "tausyn": 0.5 * ms,
    "tausyn_rise": 2 * ms
}

alpha_params_conductance = {
    "tausyn": 0.5 * ms,
    "tausyn_rise": 1 * ms
}

# Resonant kernel ##
resonant_kernel = {
    'model': """
        %kernel  = s * omega : {unit} * second **-1
        ds/dt = -s/tausyn - I_syn*omega : amp (clock-driven)

        omega: 1/second
        tausyn_kernel : second
        """,
    'on_pre': """
        s += baseweight * w_plast * abs(weight)
        %I_syn += 0 * amp
        """,
    'on_post': """ """,
}

resonant_params_current = {
    "tausyn": 0.5 * ms,
    "omega": 3 / ms,
    "tausyn_kernel": 0.5 * ms
}

resonant_params_conductance = {
    "tausyn": 0.5 * ms,
    "omega": 1 / ms,
}

none_params = {}

none_model = {
    'model': """
         """,
    'on_pre': """
         """,
    'on_post': """
         """
}
unit_less = {
    'model': """
         """,
    'on_pre': """
         """,
    'on_post': """
         """
}

""" Structural plasticity blocks:
You need to declare two set of parameters for every block:
*   current based models
*   conductance based models

These blocks add a counter that keep track of weight updates so that
we can measure how active a synapse is. Note that stochastic structural
plasticity should only be used in conjunction with synaptic plasticity
and Teili's run_regularly functions.
"""
stochastic_counter = {
    'model': """
        re_init_counter : 1
        """,
    'on_pre': """
        re_init_counter = re_init_counter + 1*int(lastspike_post!=lastspike_pre)*int(rand_int_Apre1 < Apost)*int(rand_int_Apre2 <= stdp_thres)
        """,
    'on_post': """
        re_init_counter = re_init_counter + 1*int(lastspike_post!=lastspike_pre)*int(rand_int_Apost1 < Apre)*int(rand_int_Apost2 <= stdp_thres)
        """
}

stochastic_counter_params = {
    "re_init_counter": 0
}

deterministic_counter = {
    'model': """
        re_init_counter : 1
        """,
    'on_pre': """
        re_init_counter = re_init_counter + 1*int(Apost > 0)
        """,
    'on_post': """
        re_init_counter = re_init_counter + 1*int(Apre > 0)
        """
}

deterministic_counter_params = {
    "re_init_counter": 0
}

"""Dictionary of keywords:

These dictionaries contains keyword and models and parameters names useful for the __init__ subroutine
Every new block dictionaries must be added to these definitions.
synaptic_equations is a dictionary that gathers all models and parameters.
"""
modes = {
    'current': current,
    'conductance': conductance,
    'DPI': dpi,
    'DPIShunting': dpi_shunt,
    'unit_less': unit_less,
    'quantized': quantized_stochastic
}

kernels = {
    'exponential': none_model,
    'alpha': alpha_kernel,
    'resonant': resonant_kernel
}

plasticity_models = {
    'non_plastic': none_model,
    'fusi': fusi,
    'stdp': stdp,
    'quantized_stochastic_stdp': quantized_stochastic_stdp,
    'deterministic_counter': deterministic_counter,
    'stochastic_counter': stochastic_counter
}

synaptic_equations = {
    'activity': activity,
    'stdgm': stdgm,
    'lfsr_syn': lfsr_syn
}

synaptic_equations.update(kernels)
synaptic_equations.update(plasticity_models)

# parameters dictionaries
current_parameters = {
    'current': current_params,
    'non_plastic': none_params,
    'fusi': fusi_params_current,
    'stdp': stdp_para_current,
    'exponential': none_params,
    'alpha': alpha_params_current,
    'resonant': resonant_params_current,
    'activity': none_params,
    'stdgm': none_params,
    'deterministic_counter': deterministic_counter_params,
    'stochastic_counter': stochastic_counter_params
    }

conductance_parameters = {
    'conductance': conductance_params,
    'non_plastic': none_params,
    'fusi': fusi_params_conductance,
    'stdp': stdp_para_conductance,
    'exponential': none_params,
    'alpha': alpha_params_conductance,
    'resonant': resonant_params_conductance,
    'activity': none_params,
    'stdgm': none_params,
    'deterministic_counter': deterministic_counter_params,
    'stochastic_counter': stochastic_counter_params
}

DPI_parameters = {
    'DPI': dpi_params,
    'exponential': none_params,
    'alpha': alpha_params_current,
    'non_plastic': none_params,
    'fusi': fusi_params_current,
    'stdp': stdp_para_current,
    'resonant': none_params,
    'activity': activity_params,
    'stdgm': none_params}

DPI_shunt_parameters = {
    'DPIShunting': dpi_shunt_params,
    'exponential': none_params,
    'non_plastic': none_params,
    'fusi': fusi_params_current,
    'stdp': stdp_para_current,
    'resonant': none_params,
    'alpha': none_params,
    'activity': none_params,
    'stdgm': none_params}

quantized_stochastic_parameters = {
    'quantized': quantized_stochastic_params,
    'non_plastic': none_params,
    'quantized_stochastic_stdp': quantized_stochastic_stdp_params,
    'lfsr_syn': lfsr_syn_params,
    'stochastic_counter': stochastic_counter_params}

unit_less_parameters = {
    'unit_less': none_params,
    'exponential': none_params,
    'non_plastic': none_params,
    'fusi': none_params,
    'stdp': none_params,
    'resonant': none_params,
    'alpha': none_params,
    'activity': none_params,
    'stdgm': stdgm_params}
