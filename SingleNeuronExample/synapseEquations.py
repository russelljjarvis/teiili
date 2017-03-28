# -*- coding: utf-8 -*-

from brian2 import *
#from NCSBrian2Lib.tools import *
from tools import *

def printeqDict(eqDict):
    print( 'Model equation:')
    print( eqDict['model'])
    print( '-_-_-_-_-_-_-_-')
    print( 'on pre equation:')
    print( eqDict['on_pre'])
    print( '-_-_-_-_-_-_-_-')
    if any(eqDict.keys() == 'on_post'):
        print( 'on post equation:')
        print( eqDict['on_post'])
        print( '-------------')


def MemristiveFusiSynapses(Imemthr=None, theta_dl=None, theta_du=None,
                          theta_pl=None, theta_pu=None, R_min=None, R0_d=None, R0_p=None,
                          alpha_d=None, alpha_p=None, beta_p=None, C_p=None, D_p=None, D_d=None,
                          Iw_fm=None, Iwca=None, tau_fm=None, prop=None, debug=False):

    '''
    Fusi memristive synapses

    Memristive update is based on empirical data fitting by CNR Milano on HfO thin film device.
    Depression equation: R / R0 = 1 + alpha_d * ln(n) 
    Potentiation equation: R / R0 = 1 + alpha_p * (n**-beta_p - 1)

    R0 initial resistance, n pulses nember, alpha_d, alpha_p and beta_p fitting parameters

    Equations for neurons group in input must have defined Imem and Ica variables

    All parameters below are added to the synapse model as internal variable, thus for each synapse 
    in the network a parameter is defined. Imemthr refers to the post-synaptic neuron.

    Inputs:
        load_path:          path to file txt where to load parameters, if different from default   
        Parameters:         Required parameters for fusi learning and memristive fit
                            Parameters required:
                            Imemthr:    spiking threshold for neurons
                            theta_dl:   calcium lower threshold for depression
                            theta_du:   calcium upper threshold for depression
                            theta_pl:   calcium lower threshold for potentiation
                            theta_pu:   calcium upper threshold for potentiation
                            R_min:      Minimum resistance reachable for the device
                            R0_d:       Pristine resistance for depression (average on experimental data)
                            R0_p:       Pristine resistance for potentiation (average on experimental data)
                            alpha_d:    fitting parameter depression
                            alpha_p:    fitting parameter potentiation
                            beta_p:     fitting parameter potentiation
                            C_p:        fitting parameter of value dispersion for potentiation
                            D_p:        fitting parameter of value dispersion for potentiation
                            D_d:        fitting parameter of value dispersion for depression
                            Iw_fm:      synaptic gain
                            tau_fm:     synaptic time constant
                            Iwca :      Calcium current gain

    Output: 
        Dictionary containing model, on_pre and on_post strings for synapses group
        Dictionary containing parameters

    Author: Daniele Conti 
    Author mail: daniele.conti@polito.it
    Date: 16.12.2016 
    '''

    arguments = dict(locals())

    model_fm = '''
            w : 1
            dIsyn / dt = - Isyn / tau_fm : amp (event-driven)
            Iin_ex_post = Isyn : amp (summed)
            Iwca : amp (constant)
            theta_pl : amp (constant)
            theta_dl : amp (constant)
            theta_pu : amp (constant)
            theta_du : amp (constant)
            Imemthr : amp (constant)
            Iw_fm : amp (constant)
            tau_fm : second (constant)
            prop : 1 (constant)
            R_min : 1 (constant)
            R0_d : 1 (constant)
            R0_p : 1 (constant)
            alpha_p : 1 (constant)
            alpha_d : 1 (constant)
            beta_p : 1 (constant)
            C_p : 1 (constant)
            D_p : 1 (constant)
            D_d : 1 (constant)
   
            R_new_p : 1     #for monitors only
            noiseRnewp : 1  #for monitors only
            B : 1           #for monitors only
            Rt0 : 1         #for monitors only
            '''
    on_pre_fm = '''
            up = 1. * (Imem > Imemthr) * (Ica > theta_pl) * (Ica < theta_pu)
            down = 1. * (Imem < Imemthr) * (Ica > theta_dl) * (Ica < theta_du)
                                                                                             
            Rt0 = prop * R_min / w                        
            Gt0 = 1. / Rt0

            sigma_p = C_p / (alpha_p / (Rt0 - R0_p + alpha_p)) + D_p
            noiseRnewp = sigma_p * randn()               #draw variability for the new memristor resitance
            C = alpha_p / (Rt0 - R0_p + alpha_p) + 1
            C = C * (C > 0) + 1000 * (1 - (C > 0))       #check in order to avoid negative value, 1000 is arbitrary value
            B = (alpha_p * beta_p) / C**(1 + 1 / beta_p) #if C negative the denominator becomes imaginary
            R_new_p = Rt0 - B + noiseRnewp

            R_new_d = Rt0 + alpha_d * e**( - (Rt0 - R0_d) / (alpha_d) + 1) + D_d * randn()
                              
            G_new_d = 1. / R_new_d
            G_new_p = 1. / R_new_p

            delta_G_p = G_new_p - Gt0
            delta_G_d = G_new_d - Gt0
                                
            delta_pot = prop * R_min * delta_G_p
            delta_dep = prop * R_min * delta_G_d
            w += up * delta_pot + down * delta_dep
                               
            Isyn += (Iw_fm * w) 
            '''
    on_post_fm = '''Ica += Iwca'''

    del(arguments['debug'])

    model_fm = replaceConstants(model_fm, arguments, debug)
            
    SynDict = dict(model=model_fm, on_pre=on_pre_fm, on_post=on_post_fm)

    if debug:
        printeqDict(SynDict)

    return SynDict

    #synapses group is called as follow:
    #S = Synapses(populations1, population2, model=SynDict['model'], on_pre=SynDict['on_pre'], on_post=SynDict['on_post'], method = 'euler')


def DefaultInhibitorySynapses(tauinhib=None, Iw_in_h=None, inh2output=False, debug=False):

    '''
    Default Inhibitory Synapse with current decaying in time
    Input Parameters:
        tauinhib:    synapse time constant
        Iw_in_h:     synaptic gain for synapses not going to output neurons
        Iwinh:       synaptic gain for synapses to the output neurons (substracted)

    Output: 
        Dictionary containing model, on_pre and on_post strings for synapses group

    Author: Daniele Conti 
    Author mail: daniele.conti@polito.it
    Date: 10.01.2017 
    '''
    arguments = dict(locals())

    model_inh='''
              w:1
              dIsyn/dt = (-Isyn) / tauinhib : amp (event-driven) 
              Iin_inh_post = Isyn : amp (summed)

              tauinhib : second (constant)
              Iw_in_h : amp (constant)
              Iwinh : amp (constant)
              '''
    on_pre_inh='''
              Isyn += Iw_in_h * w
              '''

    on_pre_inh_out='''
              Isyn -= Iwinh * w
              '''


    del(arguments['debug'])
    del(arguments['inh2output'])

    model_inh = replaceConstants(model_inh, arguments, debug)

    if inh2output:
        SynDict = dict(model=model_inh, on_pre=on_pre_inh_out)
    else:
        SynDict = dict(model=model_inh, on_pre=on_pre_inh)

    if debug:
        printeqDict(SynDict)

    return SynDict


def DefaultTeacherSynapses(tauexc=None, Iwexc=None, debug=False):

    '''
    Default Teacher Synapse with current decaying in time
    Input Parameters:
        tauexc:    synapse time constant
        Iwexc:     synaptic gain 

    Output: 
        Dictionary containing model, on_pre and on_post strings for synapses group

    Author: Daniele Conti 
    Author mail: daniele.conti@polito.it
    Date: 10.01.2017 
    '''
    arguments = dict(locals())

    model_teach='''
                 w : 1
                 dIsyn/dt = (Iwexc - Isyn) / tauexc : amp (event-driven) 
                 Iin_teach_post = Isyn : amp (summed)

                 tauexc : second (constant)
                 Iwexc : amp (constant)
                 '''
    on_pre_teach='''
                 Isyn += Iwexc * w
                 '''

    del(arguments['debug'])

    model_teach = replaceConstants(model_teach, arguments, debug)

    SynDict = dict(model=model_teach, on_pre=on_pre_teach)

    if debug:
        printeqDict(SynDict)

    return SynDict

def KernelsSynapses(tau=None,omega=None,sigma_gaussian=None,kernel=None,debug=False):

    '''
    Kernel Synapse function

    The kernel has to be specified when the funtion is called:
    kernel= alpha; for an alpha function
    kernel= expdecay; for exponential decay response
    kernel= gaussian; for a gaussian function
    kernel= resonant; for resosnant function

    In the Synaptic Kernel Method from Tapson et al.(2013), There is a input layer that connects to
    a hidden layer that represents the synapses, each of these synapses implements a synaptic kernel
    filter. The kernel function simply represents the response of the synapse due to an input. The three
    important properties of these filters are:
    * Convert spikes into continuos signal
    * Project the input into a higher-dimensional space
    * Implement a short-time memory of past events 

    Comments:
    * Weights are not in the default parameters.
    * t_spike must be set to 1 second as initial value to avoid kernel function at time zero (when no spike is received)
    * Equations for neurons group in input must have defined Iin_ex, in order to pass information from 
        synapse to neuron.
    * All parameters below are added to the synapse model as internal variable, thus for each synapse 
    in the network a parameter is defined.

    Inputs:
        Parameters:         Required parameters for kernel function
                            tau: time constant
                            t_spike: time when a spike occurs will be reset to zero
                            omega: for the resonant function
                            sigma: standard deviation for gaussian function


    Output: 
        Dictionary containing model and on_pre strings for synapses group
        Arguments that pass the non-default parameters

    Author: Karla Burelo 
    Date: 25.03.2017 
    '''
    arguments = dict(locals())
 
    
    

    model_alpha_fm = '''
            w : amp
            tau: second
            omega: 1/second
            sigma_gaussian : second 
            dI_alpha/dt  = -I_alpha/tau+w*exp(-t_spike/tau)/tau: amp (clock-driven)
            dt_spike/dt = 1 : second (clock-driven) 
            Iin_ex_post = I_alpha : amp (summed)
            '''
    model_resonant_fm= '''
            w : amp
            tau: second
            omega: 1/second
            sigma_gaussian : second 
            dI_resonant/dt  = (w*exp(-t_spike/tau)*cos(omega*t_spike)*omega-I_resonant/tau) : amp (clock-driven)
            dt_spike/dt = 1 : second (clock-driven) 
            Iin_ex_post = I_resonant : amp (summed)
            '''
    model_expdecay_fm= '''
            w : amp
            tau: second
            omega: 1/second
            sigma_gaussian : second 
            dI_expdecay/dt  = -I_expdecay/tau : amp (clock-driven)
            dt_spike/dt = 1 : second (clock-driven) 
            Iin_ex_post = I_expdecay : amp (summed)
            '''
    model_gaussian_fm= '''
            w : amp
            tau: second
            omega: 1/second
            sigma_gaussian : second 
            dI_gaussian/dt  = (-I_gaussian*t_spike/(sigma_gaussian**2)) :amp (clock-driven)
            dt_spike/dt = 1 : second (clock-driven) 
            Iin_ex_post = I_gaussian : amp (summed)
            '''
    on_pre_fm = '''
            t_spike = 0 * ms
            '''
    on_pre_expdecay_fm = '''
            I_expdecay +=w
            t_spike = 0 * ms
            '''
    on_pre_gaussian_fm = '''
            I_gaussian +=w
            t_spike = 0 * ms
            '''

    del(arguments['debug'])
    del(arguments['kernel'])

    #model_fm = replaceConstants(model_fm, arguments, debug)
      
    if kernel == 'alpha':
        SynDict = dict(model=model_alpha_fm, on_pre=on_pre_fm)
    elif kernel == 'resonant':
        SynDict = dict(model=model_resonant_fm, on_pre=on_pre_fm)
    elif kernel == 'expdecay':
        SynDict = dict(model=model_expdecay_fm, on_pre=on_pre_expdecay_fm)
    elif kernel == 'gaussian':
        SynDict = dict(model=model_gaussian_fm, on_pre=on_pre_gaussian_fm)
    else:
        print('Kernel not specified in the function')

    if debug:
        printeqDict(SynDict)

    return SynDict, arguments





