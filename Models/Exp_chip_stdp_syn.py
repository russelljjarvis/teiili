from NCSBrian2Lib.Parameters.Exp_chip_syn_param import parameters

model= {'model': """

            dIe_syn/dt = (-Ie_syn) / tausyne + kernel_e: amp (clock-driven)
            dIi_syn/dt = (-Ii_syn) / tausyni + kernel_i : amp (clock-driven)

            kernel_e : amp* second **-1
            kernel_i : amp* second **-1

            Ie0_post = Ie_syn : amp  (summed)
            Ii0_post = Ii_syn : amp  (summed)
            weight : 1
            wPlast : 1

            baseweight_e : amp (constant)     # synaptic gain
            baseweight_i : amp (constant)     # synaptic gain

            tausyne = Csyn * Ut_syn /(kappa_syn * Itau_e) : second
            tausyni = Csyn * Ut_syn /(kappa_syn * Itau_i) : second

            kappa_syn = (kn_syn + kp_syn) / 2 : 1

            Itau_e : amp
            Itau_i : amp

            Iw_e = weight*baseweight_e  : amp
            Iw_i = weight*baseweight_i  : amp

            Igain = Io*exp(-kappa_syn*(Vth_syn-Vdd_syn)/Ut_syn) : amp

            duration_syn : second (constant)
            kn_syn       : 1 (constant)
            kp_syn       : 1 (constant)
            Ut_syn       : volt (constant)
            Io_syn       : amp (constant)
            Csyn         : farad (constant)
            Vdd_syn      : volt (constant)
            Vth_syn      : volt (constant)
                 

            w : 1
            dApre/dt = -Apre / taupre : 1 (event-driven)
            dApost/dt = -Apost / taupost : 1 (event-driven)
            w_max: 1 (shared, constant)
            taupre : second (shared, constant)
            taupost : second (shared, constant)
            diffApre : 1 (shared, constant)
            Q_diffAPrePost : 1 (shared, constant)
                        
            """,
            
        'on_pre': """            
                 Ie_syn += Iw_e*Igain*wPlast*(weight>0)/Itau_e
                 Ii_syn += Iw_i*Igain*wPlast*(weight<0)/Itau_i
                 t_spike = 0 * ms
                  
                 wPlast = w
                 Apre += diffApre*w_max
                 w = clip(w + Apost, 0, w_max) 
                 """,
                 
        'on_post': """      
        Apost += -diffApre * (taupre / taupost) * Q_diffAPrePost * w_max
        w = clip(w + Apre, 0, w_max) 
        """,
        
        'parameters':parameters
        }

