from NCSBrian2Lib.Parameters.Exp_chip_syn_param import parameters

Exp_chip_syn = {'model': """
            dIe_syn/dt = (-Ie_syn) / tausyne + kernel_e: amp (clock-driven)
            dIi_syn/dt = (-Ii_syn) / tausyni + kernel_i : amp (clock-driven)

            %kernel_e = -{synvar_e}**2/(Igain*tausyne) + Igain*{synvar_e}*(Iw_e-Itau_e)/(tausyne*Itau_e*(Igain + {synvar_e})) : amp * second **-1
            %kernel_i = +{synvar_i}**2/(Igain*tausyni) + Igain*{synvar_i}*(Iw_i+Itau_i)/(tausyni*Itau_i*(-Igain + {synvar_i})) : amp * second **-1


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

            Igain : amp

            duration_syn : second (constant)
            kn_syn       : 1 (constant)
            kp_syn       : 1 (constant)
            Ut_syn       : volt (constant)
            Io_syn       : amp (constant)
            Csyn         : farad (constant)
            Vdd_syn      : volt (constant)
            Vth_syn      : volt (constant)                 
            """,
        'on_pre': """            
                 Ie_syn += Iw_e*Igain*wPlast*(weight>0)/Itau_e
                 Ii_syn += Iw_i*Igain*wPlast*(weight<0)/Itau_i
                 t_spike = 0 * ms
                  """,
        'on_post': """ """,
        'parameters':parameters
        }

