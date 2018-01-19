# -*- coding: utf-8 -*-
# @Author: mrax, mmilde
# @Date:   2017-12-27 10:46:44
# @Last Modified by:   mmilde
# @Last Modified time: 2018-01-17 15:43:31

"""File contains a pre-defined DPI synapse model for the somatic inhibitory synapse (shunting), with all its attributes required by brian2.
This model was orinally pubished in Chicca et al. 2014

Attributes:
    dpi_shunt_syn_eq (dict): Dictionary which specifies shunting inhibitory synapse according to Differential Pair Intgrator (DPI)
        synapse circuit.
"""


from NCSBrian2Lib.Parameters.Shunt_syn_param import parameters

dpi_shunt_syn_eq = {'model': """
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
