from NCSBrian2Lib.Parameters.Exp_syn_param import parameters

Exp_syn = {'model': """
            dIe_syn/dt = (-Ie_syn) / tausyne + kernel_e: amp (clock-driven)
            dIi_syn/dt = (-Ii_syn) / tausyni + kernel_i : amp (clock-driven)

            kernel_e : amp* second **-1
            kernel_i : amp* second **-1

            Ie0_post = Ie_syn : amp  (summed)
            Ii0_post = Ii_syn : amp  (summed)
            weight : 1
            tausyne : second (constant) # synapse time constant
            tausyni : second (constant) # synapse time constant
            wPlast : 1

            baseweight_e : amp (constant)     # synaptic gain
            baseweight_i : amp (constant)     # synaptic gain""",
        'on_pre':""" 
            Ie_syn += baseweight_e * weight *wPlast*(weight>0)
            Ii_syn += baseweight_i * weight *wPlast*(weight<0)""",
        'on_post': """ """,
        'parameters':parameters
        }

