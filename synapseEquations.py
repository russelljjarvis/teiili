# -*- coding: utf-8 -*-

from brian2 import *
from NCSBrian2Lib.tools import *

def printeqDict(eqDict):
    print( 'Model equation:')
    print( eqDict['model'])
    print( '-_-_-_-_-_-_-_-')
    print( 'on pre equation:')
    print( eqDict['on_pre'])
    print( '-_-_-_-_-_-_-_-')
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
            '''
    on_pre_fm = '''
            up = 1. * (Imem > Imemthr) * (Ica > theta_pl) * (Ica < theta_pu)
            down = 1. * (Imem < Imemthr) * (Ica > theta_dl) * (Ica < theta_du)
                                                                                             
            Rt0 = prop * R_min / w                        
            Gt0 = 1. / Rt0

            sigma_p = C_p / (alpha_p / (Rt0 - R0_p + alpha_p)) + D_p
            R_new_p = Rt0 - (alpha_p * beta_p) / ((alpha_p / (Rt0 - R0_p + alpha_p) + 1)**(1 + 1 / beta_p)) #+ sigma_p * randn()
            R_new_d = Rt0 + alpha_d * e**( - (Rt0 - R0_d) / (alpha_d) + 1) #+ D_d * randn()
                              
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









