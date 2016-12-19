# -*- coding: utf-8 -*-

from brian2 import *


def MemristiveFusiSynapses(load_path=None, Imemthr=None, theta_dl=None, theta_du=None,
                          theta_pl=None, theta_pu=None, R_min=None, R0_d=None, R0_p=None,
                          alpha_d=None, alpha_p=None, beta_p=None, C_p=None, D_p=None, D_d=None,
                          Iw_fm=None, Iwca=None, tau_fm=None, prop=None):

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

    default_load_path = '../SynapsesParamters/memfusi_parameters.txt'

    Pdict = {}
    if load_path is not None:
        path = load_path
    else:
        path = default_load_path

    PFile = open(path, 'r')
    for line in PFile:
        (key, trash, val) = line.split()
        Pdict[key] = val
    PFile.close()


    model_fm = '''
            w : 1
            dIsyn / dt = - Isyn / tau_fm : amp (event-driven)
            Iin_post = Isyn : amp (summed)
            '''
    on_pre_fm = '''
            up = 1. * (Imem > Imemthr) * (Ica > theta_pl) * (Ica < theta_pu)
            down = 1. * (Imem < Imemthr) * (Ica > theta_pl) * (Ica < theta_du)
                                                                                             
            Rt0 = prop * R_min / w                        
            Gt0 = 1. / Rt0

            sigma_p = C_p / (alpha_p / (Rt0 - R0_p + alpha_p)) + D_p
            R_new_p = Rt0 - alpha_p * beta_p / (alpha_p / (Rt0 - R0_p + alpha_p) + 1)**(1 + 1 / beta_p) + sigma_p * randn()
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

    del(arguments['load_path'])


    #check if among input to function some parameter has been specified and updating parameters dictionary
    for key in arguments:
        if arguments[key] is not None:
            if key in Pdict.keys():
                #input variable is <class 'brian2.units.fundamentalunits.Quantity'> and not a string
                #it is acquired as 1. nA and transformed into a string '1.*nA', then added to dict
                tmp = str(arguments[key])
                tmp = tmp.split()
                new_val = tmp[0] + '*' + tmp[1]
                Pdict[key] = new_val

    #adding parameters to the synapse model with the correct unit of measure
    for key in Pdict:
        if ('nA' in Pdict[key]) or ('namp' in Pdict[key]) or ('pA' in Pdict[key]) or ('pamp' in Pdict[key]):
            model_fm += key + ''' = ''' + Pdict[key] + ''' : amp (constant over dt)
             '''
        elif ('ms' in Pdict[key]) or ('msec' in Pdict[key]) or ('msecond' in Pdict[key]):
            model_fm += key + ''' = ''' + Pdict[key] + ''' : second (constant over dt)
             '''
        else:
            model_fm += key + ''' = ''' + Pdict[key] + ''' : 1 (constant over dt)
             '''

    SynDict = dict(model=model_fm, on_pre=on_pre_fm, on_post=on_post_fm)

    return SynDict, Pdict

    #synapses group is called as follow:
    #S = Synapses(populations1, population2, model=SynDict['model'], on_pre=SynDict['on_pre'], on_post=SynDict['on_post'], method = 'euler')









