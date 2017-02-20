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
    if any(eqDict.keys() == 'on_post'):
        print( 'on post equation:')
        print( eqDict['on_post'])
        print( '-------------')


def MemristiveFusiSynapses(Imemthr=None, theta_dl=None, theta_du=None,
                          theta_pl=None, theta_pu=None, R_min=None, R0_d=None, R0_p=None,
                          alpha_d=None, alpha_p=None, beta_p=None, C_p=None, D_p=None, D_d=None,
                          Iw_fm=None, Iwca=None, tau_fm=None, prop=None, debug=False, plastic=True):

    '''
    Fusi memristive synapses

    if plastic=True the resistance update is executed

    Memristive update is based on empirical data fitting by CNR Milano on HfO thin film device.
    Depression equation: R / R0 = 1 + alpha_d * ln(n) 
    Potentiation equation: R / R0 = 1 + alpha_p * (n**-beta_p - 1)

    R0 initial resistance, n pulses nember, alpha_d, alpha_p and beta_p fitting parameters

    Equations for neurons group in input must have defined Imem and Ica variables

    All parameters below are added to the synapse model as internal variable, thus for each synapse 
    in the network a parameter is defined. Imemthr refers to the post-synaptic neuron.

    Inputs:
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
            R_max : 1 (constant)
            R0_d : 1 (constant)
            R0_p : 1 (constant)
            alpha_p : 1 (constant)
            alpha_d : 1 (constant)
            beta_p : 1 (constant)
            C_p : 1 (constant)
            D_p : 1 (constant)
            D_d : 1 (constant)
            count_up : 1 (constant)
            count_down : 1 (constant)
            '''
    on_pre_fm = '''
            up = 1. * (Imem > Imemthr) * (Ica > theta_pl) * (Ica < theta_pu)
            down = 1. * (Imem < Imemthr) * (Ica > theta_dl) * (Ica < theta_du)
                                                                                             
            Rt0 = prop * R_min / w  #Rt0 = prop * R_min * R_max / (w * R_max + prop * R_min)  equivalent to w = prop * (R_min / R - R_min / R_max)                
            Gt0 = 1. / Rt0

            sigma_p = C_p / (alpha_p / (Rt0 - R0_p + alpha_p)) + D_p
            noiseRnewp = sigma_p * randn()               #draw variability for the new memristor resitance
            C = alpha_p / (Rt0 - R0_p + alpha_p) + 1
            C = C * (C > 0) + 1000 * (1 - (C > 0))       #check in order to avoid negative value, 1000 is arbitrary value
            B = (alpha_p * beta_p) / C**(1 + 1 / beta_p) #if C negative the denominator becomes imaginary
            R_new_p = Rt0 - B #+ noiseRnewp            

            R_new_d = Rt0 + alpha_d * e**( - (Rt0 - R0_d) / (alpha_d) + 1) #+ D_d * randn()
                              
            G_new_d = 1. / R_new_d
            G_new_p = 1. / R_new_p
                        
            delta_G_p = G_new_p - Gt0
            delta_G_d = G_new_d - Gt0
                                
            delta_pot = prop * R_min * delta_G_p
            delta_dep = prop * R_min * delta_G_d
            w += up * delta_pot + down * delta_dep
            count_up += up
            count_down += down
                   
            Isyn += Iw_fm * w 
            '''

    on_pre_fm_nonplastic='''
            Isyn += Iw_fm * w
            '''

    on_post_fm = '''Ica += Iwca'''

    del(arguments['debug'])
    del(arguments['plastic'])

    if plastic:
        SynDict = dict(model=model_fm, on_pre=on_pre_fm, on_post=on_post_fm)
    else:
        SynDict = dict(model=model_fm, on_pre=on_pre_fm_nonplastic, on_post=on_post_fm)
        
    if debug:
        printeqDict(SynDict)

    return SynDict, arguments

    #synapses group is called as follow:
    #S = Synapses(populations1, population2, method = 'euler', **SynDict)

def DefaultExcitatorySynapses(tauexc=None, Iw_exc=None, debug=False):
    
    '''
    Default Excitatory Synapse with current decaying in time
    Input Parameters:
        tauexc:    synapse time constant
        Iw_exc:     synaptic gain 

    Output: 
        Dictionary containing model, on_pre and on_post strings for synapses group

    Author: Daniele Conti 
    Author mail: daniele.conti@polito.it
    Date: 27.01.2017 
    '''
    arguments = dict(locals())

    model_ex='''
                 w : 1
                 dIsyn_exc/dt = (-Isyn_exc) / tauexc : amp (event-driven) 
                 Iin_ex_post = Isyn_exc : amp (summed)

                 tauexc : second (constant)
                 Iw_exc : amp (constant)
                 '''
    on_pre_ex='''
                 Isyn_exc += Iw_exc * w
                 '''

    del(arguments['debug'])

    SynDict = dict(model=model_ex, on_pre=on_pre_ex)

    if debug:
        printeqDict(SynDict)

    return SynDict, arguments


def DefaultInhibitorySynapses(tauinhib=None, Iw_inh=None, inh2output=False, debug=False):
    
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
              dIsyn_inh/dt = (-Isyn_inh) / tauinhib : amp (event-driven) 
              Iin_inh_post = Isyn_inh : amp (summed)

              tauinhib : second (constant)
              Iw_inh : amp (constant)
              '''
    on_pre_inh='''
              Isyn_inh += Iw_inh * w
              '''

    on_pre_inh_out='''
              Isyn_inh -= Iw_inh * w
              '''


    del(arguments['debug'])
    del(arguments['inh2output'])

    if inh2output:
        SynDict = dict(model=model_inh, on_pre=on_pre_inh_out)
    else:
        SynDict = dict(model=model_inh, on_pre=on_pre_inh)

    if debug:
        printeqDict(SynDict)

    return SynDict, arguments




def DefaultTeacherSynapses(taut=None, Iw_t=None, debug=False):
    
    '''
    Default Teacher Synapse with current decaying in time
    Input Parameters:
        taut:    synapse time constant
        Iw_t:     synaptic gain 

    Output: 
        Dictionary containing model, on_pre and on_post strings for synapses group

    Author: Daniele Conti 
    Author mail: daniele.conti@polito.it
    Date: 10.01.2017 
    '''
    arguments = dict(locals())

    model_teach='''
                 w : 1
                 dIsyn_teach/dt = (-Isyn_teach) / taut : amp (event-driven) 
                 Iin_teach_post = Isyn_teach : amp (summed)

                 taut : second (constant)
                 Iw_t : amp (constant)
                 '''
    on_pre_teach='''
                 Isyn_teach += Iw_t * w
                 '''

    del(arguments['debug'])

    SynDict = dict(model=model_teach, on_pre=on_pre_teach)

    if debug:
        printeqDict(SynDict)

    return SynDict, arguments


def reversalSyn(taugIe = None, taugIi = None, EIe = None, EIi = None, debug=False):
    
    
    arguments = dict(locals())
    del(arguments['debug'])
    
    preEq = '''gIe += weight*nS*(weight>0)
gIi += weight*(weight<0)*nS''' 
    modelEq = '''dgIe/dt = (-gIe/taugIe) : siemens (clock-driven) # instantaneous rise, exponential decay
dgIi/dt = (-gIi/taugIi) : siemens  (clock-driven) # instantaneous rise, exponential decay
Ies = gIe*(EIe - Vm_post) :amp
Iis = gIi*(EIi - Vm_post) :amp
taugIe : second (constant)        # excitatory input time constant
taugIi : second (constant)        # inhibitory input time constant
EIe : volt (constant)             # excitatory reversal potential
EIi : volt (constant)             # inhibitory reversal potential
weight : 1 (constant)
Vm_post : volt
Ie_post = Ies : amp  (summed)
Ii_post = Iis : amp  (summed)
'''
    
    modelEq = replaceConstants(modelEq,arguments,debug)
    preEq = replaceConstants(preEq,arguments,debug)
    
    SynDict = dict(model=modelEq, on_pre=preEq)
   
    if debug:
        print('arguments of ExpAdaptIF: \n' + str(arguments))
        printeqDict(SynDict)

    return SynDict


def fusiSyn(taugIe = None, EIe = None, w_plus = None, w_minus= None,
            theta_upl= None, theta_uph= None, theta_downh = None, theta_downl = None,
            theta_V = None, alpha = None, beta = None, tau_ca = None, w_ca = None, debug = False):         
    ''' This is still not completely tested and might contain bugs.
    '''
    arguments = dict(locals())
    del(arguments['debug'])

    modelEq = '''dgIe/dt = (-gIe/taugIe) : siemens (clock-driven) # instantaneous rise, exponential decay
Ies = gIe*(EIe - Vm_post) :amp
taugIe : second (constant)        # excitatory input time constant
EIe : volt (constant)             # excitatory reversal potential
dCa/dt = (-Ca/tau_ca) : volt (clock-driven) #Calcium Potential
dw/dt = (alpha*(w>theta_w)*(w<w_max))-(beta*(w<=theta_w)*(w>w_min)) : 1 (clock-driven) # internal weight variable
w_plus: 1 (constant)             
w_minus: 1 (constant)
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
w_ca: volt (constant)            # Calcium weight
Vm_post : volt
Ie_post = Ies : amp  (summed)
weight: 1 (constant)
'''
    preEq  = '''
gIe += floor(w+0.5) * weight *  nS
w += w_plus  * (Vm_post>theta_V) * (Ca>theta_upl)   * (Ca<theta_uph)   *(w<w_max) 
w -= w_minus * (Vm_post<theta_V) * (Ca>theta_downl) * (Ca<theta_downh) *(w>w_min)
''' #  check if correct
    postEq  = '''Ca += w_ca '''

      
    modelEq = replaceConstants(modelEq,arguments,debug)
    preEq   = replaceConstants(preEq,arguments,debug)
    postEq  = replaceConstants(postEq,arguments,debug)
    
    SynDict = dict(model=modelEq, on_pre=preEq, on_post=postEq)
   
    if debug:
        print('arguments of ExpAdaptIF: \n' + str(arguments))
        printeqDict(SynDict)

    return SynDict


def BraderFusiSynapses(Imemthr=None, theta_dl=None, theta_du=None,
                       theta_pl=None, theta_pu=None,
                       Iw_fm=None, Iwca=None, tau_fm=None, prop=None, debug=False, plastic=True):

    '''
    Fusi memristive synapses

    if plastic=True the resistance update is executed

    Equations for neurons group in input must have defined Imem and Ica variables

    All parameters below are added to the synapse model as internal variable, thus for each synapse 
    in the network a parameter is defined. Imemthr refers to the post-synaptic neuron.

    Inputs:
        Parameters:         Required parameters for fusi learning and memristive fit
                            Parameters required:
                            Imemthr:    spiking threshold for neurons
                            theta_dl:   calcium lower threshold for depression
                            theta_du:   calcium upper threshold for depression
                            theta_pl:   calcium lower threshold for potentiation
                            theta_pu:   calcium upper threshold for potentiation
                            Iw_fm:      synaptic gain
                            tau_fm:     synaptic time constant
                            Iwca :      Calcium current gain

    Output: 
        Dictionary containing model, on_pre and on_post strings for synapses group

    Author: Daniele Conti 
    Author mail: daniele.conti@polito.it
    Date: 20.02.2017 
    '''

    arguments = dict(locals())

    model_fm = '''
            dIsyn / dt = - Isyn / tau_fm : amp (event-driven)
            dw / dt = alpha * (w > wth) - beta * (w < wth) : 1 (clock-driven)

            Iin_ex_post = Isyn : amp (summed)
            Iwca : amp (constant)
            theta_pl : amp (constant)
            theta_dl : amp (constant)
            theta_pu : amp (constant)
            theta_du : amp (constant)
            Imemthr : amp (constant)
            Iw_fm : amp (constant)
            tau_fm : second (constant)
            alpha : 1 (constant)
            beta : 1 (constant)
            wth : 1 (constant)
            count_up : 1 (constant)
            count_down : 1 (constant)
            '''
    on_pre_fm = '''
            up = 1. * (Imem > Imemthr) * (Ica > theta_pl) * (Ica < theta_pu)
            down = 1. * (Imem < Imemthr) * (Ica > theta_dl) * (Ica < theta_du)
                                                                                             
            w += up * alpha - down * beta
            count_up += up
            count_down += down
            
            w = clip(w,0,1)       
            Isyn += Iw_fm * w 
            '''

    on_pre_fm_nonplastic='''
            Isyn += Iw_fm * w
            '''

    on_post_fm = '''Ica += Iwca'''

    del(arguments['debug'])
    del(arguments['plastic'])

    if plastic:
        SynDict = dict(model=model_fm, on_pre=on_pre_fm, on_post=on_post_fm)
    else:
        SynDict = dict(model=model_fm, on_pre=on_pre_fm_nonplastic, on_post=on_post_fm)
        
    if debug:
        printeqDict(SynDict)

    return SynDict, arguments

    #synapses group is called as follow:
    #S = Synapses(populations1, population2, method = 'euler', **SynDict)
