# -*- coding: utf-8 -*-

from brian2 import *
#from NCSBrian2Lib.tools import *

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


def DefaultExcitatorySynapses(tauexc=None, Iw_exc=None, debug=False):

    '''
    Default Excitatory Synapse with exponentially decaying current in time
    Input Parameters:
        tauexc:    synapse time constant
        Iw_exc:    synaptic gain

    Output:
        Dictionary containing model, on_pre and on_post strings for synapses group, dictionary of non-default parameters

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


def DefaultInhibitorySynapses(tauinhib=None, Iw_inh=None, inh2output=True, debug=False):

    '''
    Default Inhibitory Synapse with current decaying in time
    Input Parameters:
        tauinhib:    synapse time constant
        Iw_inh:      synaptic gain (substracted only for synapses to output neurons)

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
    Default Teacher Synapse with exponentially decaying current in time
    Input Parameters:
        taut:    synapse time constant
        Iw_t:    synaptic gain

    Output:
        Dictionary containing model, on_pre and on_post strings for synapses group, dictionary of non-default parameters

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

def simpleSyn(inputNumber = 1, debug=False):
    '''
    simple excitatory or inhibitory Synapse with instantaneous rise - exponential decay kernel
    Input Arguments:
        Ie can be Ie or Ie2, Ie3, ... (if there are several Inputs to a post neuron)
        Ii can be Ii or Ii2, ...
        debug determines if the equation is printed
    Date: 03.2017
    '''
    arguments = dict(locals())
    del(arguments['debug'])


    modelEq='''
            dIe_syn/dt = (-Ie_syn) / tau : amp (clock-driven)
            dIi_syn/dt = (-Ii_syn) / tau : amp (clock-driven)
            {Ie}_post = Ie_syn : amp  (summed)
            {Ii}_post = Ii_syn : amp  (summed)
            weight : 1
            tau : second (constant) # synapse time constant
            Iw : amp (constant)     # synaptic gain
            '''
    if inputNumber > 1 :
        modelEq = modelEq.format(Ie="Ie"+str(inputNumber),Ii="Ii"+str(inputNumber))        
    else:
        modelEq = modelEq.format(Ie="Ie",Ii="Ii")  
    
    preEq='''
        Ie_syn += Iw * weight *(weight>0)
        Ii_syn += Iw * weight * (weight<0)
        '''

    SynDict = dict(model=modelEq, on_pre=preEq)

    if debug:
        print('arguments of ExpAdaptIF: \n' + str(arguments))
        printeqDict(SynDict)

    return SynDict


def reversalSynV(inputNumber = 1, debug=False):
    '''
    Synapse with reversal potential and instantaneous rise - exponential decay kernel
    for voltage based neurons!
    Input Arguments:
        Ie can be Ie or Ie2, Ie3, ... (if there are several Inputs to a post neuron)
        Ii can be Ii or Ii2, ...
        debug determines if the equation is printed
    Author: Alpha Renner
    Date: 03.2017
    '''

    arguments = dict(locals())
    del(arguments['debug'])

    preEq = '''
            gIe += weight*gWe*(weight>0)
            gIi += weight*gWi*(weight<0)
            '''

    modelEq = '''
            dgIe/dt = (-gIe/taugIe) : siemens (clock-driven) # exponential decay
            dgIi/dt = (-gIi/taugIi) : siemens  (clock-driven) # exponential decay
            Iesyn = gIe*(EIe - Vm_post) :amp
            Iisyn = gIi*(EIi - Vm_post) :amp
            taugIe : second (constant)        # excitatory input time constant
            taugIi : second (constant)        # inhibitory input time constant
            EIe : volt (constant)             # excitatory reversal potential
            EIi : volt (constant)             # inhibitory reversal potential
            gWe : siemens (constant)          # excitatory synaptic gain
            gWi : siemens (constant)          # inhibitory synaptic gain
            weight : 1 (constant)
            {Ie}_post = Iesyn : amp  (summed)
            {Ii}_post = Iisyn : amp  (summed)
                '''
    if inputNumber > 1 :
        modelEq = modelEq.format(Ie="Ie"+str(inputNumber),Ii="Ii"+str(inputNumber))        
    else:
        modelEq = modelEq.format(Ie="Ie",Ii="Ii")

    SynDict = dict(model=modelEq, on_pre=preEq)

    if debug:
        print('arguments of reversalSynV: \n' + str(arguments))
        printeqDict(SynDict)

    return SynDict



def fusiSynV(inputNumber = 1, debug = False):
    ''' This is not yet completely tested and might contain bugs.
    '''
    arguments = dict(locals())
    del(arguments['debug'])

    modelEq = '''dgIe/dt = (-gIe/taugIe) : siemens (clock-driven)               # instantaneous rise, exponential decay
                Ies = gIe*(EIe - Vm_post) :amp
                taugIe : second (constant)                                      # excitatory input time constant
                EIe : volt (constant)                                           # excitatory reversal potential
                dCa/dt = (-Ca/tau_ca) : volt (event-driven)                     #Calcium Potential
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
                w_ca: volt (constant)                                           # Calcium weight
                {Ie}_post = Ies : amp  (summed)
                weight: 1 (constant)
                '''
    if inputNumber > 1 :
        modelEq = modelEq.format(Ie="Ie"+str(inputNumber))        
    else:
        modelEq = modelEq.format(Ie="Ie")
        
    preEq = '''
            gIe += floor(w+0.5) * weight *  nS
            w += w_plus  * (Vm_post>theta_V) * (Ca>theta_upl)   * (Ca<theta_uph)   #*(w<w_max)
            w -= w_minus * (Vm_post<theta_V) * (Ca>theta_downl) * (Ca<theta_downh) #*(w>w_min)
            w = clip(w,w_min,w_max)
            '''  #  check if correct
    postEq = '''Ca += w_ca'''

    SynDict = dict(model=modelEq, on_pre=preEq, on_post=postEq)

    if debug:
        print('arguments of ExpAdaptIF: \n' + str(arguments))
        printeqDict(SynDict)

    return SynDict

def BraderFusiSynapses(Imemthr=None, theta_dl=None, theta_du=None,
                       theta_pl=None, theta_pu=None,
                       Iw_fm=None, Iwca=None, tau_fm=None, prop=None, debug=False, plastic=True):

    '''
    Fusi bistable synapses, cfr Brader et al 2007

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
                            alpha:      Angular coefficient of increasing synaptic weight drift
                            beta:       Angular coefficient of decreasing synaptic weight drift
                            wa:         Unitary increment of synaptic weight during LTP
                            wb:         Unitary decrement of synaptic weight during LTD
                            wth:        Synaptic weight threshold for drift
                            count_up:   Auxiliary variable counting LTP transitions
                            count_down: Auxiliary variable counting LTD transitions

    Output:
        Dictionary containing model, on_pre and on_post strings for synapses group, dictionary of non-default parameters

    Author: Daniele Conti
    Author mail: daniele.conti@polito.it
    Date: 20.02.2017
    '''

    arguments = dict(locals())

    model_fm = '''
            dIsyn / dt = - Isyn / tau_fm : amp (event-driven)
            dw / dt = alpha * (w > wth) * (w < 1) - beta * (w <= wth) * (w > 0) : 1 (clock-driven)

            Iin_ex_post = Isyn : amp (summed)
            Iwca : amp (constant)
            theta_pl : amp (constant)
            theta_dl : amp (constant)
            theta_pu : amp (constant)
            theta_du : amp (constant)
            Imemthr : amp (constant)
            Iw_fm : amp (constant)
            tau_fm : second (constant)
            alpha : hertz (constant)
            beta : hertz (constant)
            wa : 1 (constant)
            wb : 1 (constant)
            wth : 1 (constant)
            count_up : 1 (constant)
            count_down : 1 (constant)
            '''
    on_pre_fm = '''
            up = 1. * (Imem > Imemthr) * (Ica > theta_pl) * (Ica < theta_pu)
            down = 1. * (Imem < Imemthr) * (Ica > theta_dl) * (Ica < theta_du)

            w += up * wa - down * wb
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
	
	alpha synapse changed by Alpha on 06.07.2017 (w-->max EPSC)
    '''
    arguments = dict(locals())

	# in the alpha snyapse, w detemines the maximal amplitude of an EPSC
    model_alpha_fm = '''
            w : amp
            tau: second
            omega: 1/second
            sigma_gaussian : second
            dI_alpha/dt  = -I_alpha/tau+w*exp(1-t_spike/tau)/tau: amp (clock-driven)
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

def SiliconSynapses(Vth=None, Vtau=None, Vdd=None,Csyn=None, Io=None,
                    Ut=None, kn=None,kp=None, duration=None,debug=False):

    '''
    Comments:
    * Weights are not in the default parameters, it must be set by the user
    and it is the current injected to the synpase when a spike is received

    Author: Karla Burelo
    Author mail: burelo.rodriguez.karla@outlook.com
    Date: 2.04.2017
    '''

    arguments = dict(locals())

    model_fm = '''
            dIsyn / dt = - Isyn / tau * (1-Iin_ex/Itau) : amp (event-driven)
            tau = Csyn * kappa /(Ut * Itau) : 1/second (clock-driven)
            Itau = Io * exp((-kappa * Vtau + Vdd)/Ut) : amp
            kappa = (kn + kp) / 2 : 1

            Iw = w*(t_spike<duration) + 0*(t_spike>= duration) : amp
            Iin_ex = Iw / (1+(Isyn/Igain)): amp
            Igain = Io*exp(-kappa*(Vth-Vdd)/Ut) : amp
            dt_spike/dt = 1 : second (clock-driven)

            w      : amp (constant)
            duration:1 (constant)
            kn     : 1 (constant)
            kp     : 1 (constant)
            Ut     : volt (constant)
            Io     : amp (constant)
            Csyn   : farad (constant)
            Vdd    : volt (constant)
            Vtau   : volt (constant)
            Vth    : volt (constant)
            '''
    on_pre_fm = '''
            t_spike = 0 * ms
            '''

    del(arguments['debug'])

    SynDict = dict(model=model_fm, on_pre=on_pre_fm)

    if debug:
        printeqDict(SynDict)

    return SynDict, arguments

    #synapses group is called as follow:
    #S = Synapses(populations1, population2, method = 'euler', **SynDict)


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
                            R_max:      Maximum resistance reachable for the device
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
                            Iwca:       Calcium current gain
                            count_up:   Auxiliary variable counting LTD transitions
                            count_down: Auxiliary variable counting LTP transitions

    Output:
        Dictionary containing model, on_pre and on_post strings for synapses group, dictionary of non-default parameters

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

            Rt0 = prop * R_min * R_max / (w * R_max + prop * R_min) # equivalent to w = prop * (R_min / R - R_min / R_max)  Rt0 = prop * R_min / w
            Gt0 = 1. / Rt0

            sigma_p = C_p / (alpha_p / (Rt0 - R0_p + alpha_p)) + D_p
            noiseRnewp = sigma_p * randn()               #draw variability for the new memristor resitance
            C = alpha_p / (Rt0 - R0_p + alpha_p) + 1
            C = C * (C > 0) + 1000 * (1 - (C > 0))       #check in order to avoid negative value, 1000 is arbitrary value and tuned further
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



def StdpSynV(inputNumber = 1, debug = False):
    ''' This an STDP synapse adapted from http://brian2.readthedocs.io/en/latest/examples/synapses.STDP.html
        after Song, Miller and Abbott (2000) and Song and Abbott (2001)
    '''

    arguments = dict(locals())
    del(arguments['debug'])

    modelEq = '''
            dgIe/dt = (-gIe/taugIe) : siemens (clock-driven) # instantaneous rise, exponential decay
            Ies = gIe*(EIe - Vm_post) :amp
            taugIe : second (constant)        # excitatory input time constant
            EIe : volt (constant)             # excitatory reversal potential
            w : 1
            weight: 1 (constant)
            dApre/dt = -Apre / taupre : 1 (event-driven)
            dApost/dt = -Apost / taupost : 1 (event-driven)
            w_max: 1 (constant)
            taupre : second (constant)
            taupost : second (constant)
            diffApre : 1 (constant)
            Q_diffAPrePost : 1 (constant)
            {Ie}_post = Ies : amp  (summed)
            '''

    if inputNumber > 1 :
        modelEq = modelEq.format(Ie="Ie"+str(inputNumber))        
    else:
        modelEq = modelEq.format(Ie="Ie")
        
    preEq = '''
            gIe += w * weight * nS
            Apre += diffApre*w_max
            w = clip(w + Apost, 0, w_max)
            '''
    postEq = '''
            Apost += -diffApre * (taupre / taupost) * Q_diffAPrePost * w_max
            w = clip(w + Apre, 0, w_max)
            '''

    SynDict = dict(model=modelEq, on_pre=preEq, on_post=postEq)

    if debug:
        print('arguments of ExpAdaptIF: \n' + str(arguments))
        printeqDict(SynDict)

    return SynDict
