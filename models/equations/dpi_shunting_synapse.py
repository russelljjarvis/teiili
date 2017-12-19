from NCSBrian2Lib.Parameters.Shunt_syn_param import parameters

Shunt_syn = {'model': """
     	       dIi_syn/dt = (-Ii_syn - Ii_gain + 2*Io_syn*(Ii_syn<=Io_syn))/(tausyni*((Ii_gain/Ii_syn)+1)) : amp (clock-driven)

            Ishunt_post = -Ii_syn : amp  (summed)

            weight : 1
            wPlast : 1
            
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
            Vdd_syn      : volt (constant)
            Vth_syn      : volt (constant)
            """,
            'on_pre': """
             Ii_syn += Iw_i*Ii_gain*(weight<0)/(Itau_i*((Ii_gain/Ii_syn)+1))
              """,
            'on_post': """ """,
            'parameters': parameters
                }
