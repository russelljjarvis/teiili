# -*- coding: utf-8 -*-
# @Author: mrax, mmilde
# @Date:   2017-12-27 10:46:44
# @Last Modified by:   mmilde
# @Last Modified time: 2018-01-17 15:45:15

"""File contains a pre-defined DPI synapse model for the excitatory and inhibitory synapse
(both dendritic), with all its attributes required by brian2. This model was orinally pubished in Chicca et al. 2014

Attributes:
    dpi_syn_eq (dict): Dictionary which specifies excitatory or inhibitory synapse according to Differential Pair Intgrator (DPI)
        synapse circuit.
"""

dpi_syn_eq = {'model': """
            dIe_syn/dt = (-Ie_syn - Ie_gain + 2*Io_syn*(Ie_syn<=Io_syn))/(tausyne*((Ie_gain/Ie_syn)+1)) : amp (clock-driven)
            dIi_syn/dt = (-Ii_syn - Ii_gain + 2*Io_syn*(Ii_syn<=Io_syn))/(tausyni*((Ii_gain/Ii_syn)+1)) : amp (clock-driven)

            {Ie}_post = Ie_syn : amp  (summed)
            {Ii}_post = -Ii_syn : amp  (summed)

            weight : 1
            wPlast : 1

            Ie_gain = Io_syn*(Ie_syn<=Io_syn) + Ie_th*(Ie_syn>Io_syn) : amp
            Ii_gain = Io_syn*(Ii_syn<=Io_syn) + Ii_th*(Ii_syn>Io_syn) : amp

            Itau_e = Io_syn*(Ie_syn<=Io_syn) + Ie_tau*(Ie_syn>Io_syn) : amp
            Itau_i = Io_syn*(Ii_syn<=Io_syn) + Ii_tau*(Ii_syn>Io_syn) : amp

            baseweight_e : amp (constant)     # synaptic gain
            baseweight_i : amp (constant)     # synaptic gain
            tausyne = Csyn * Ut_syn /(kappa_syn * Itau_e) : second
            tausyni = Csyn * Ut_syn /(kappa_syn * Itau_i) : second
            kappa_syn = (kn_syn + kp_syn) / 2 : 1


            Iw_e = weight*baseweight_e  : amp
            Iw_i = -weight*baseweight_i  : amp

            Ie_tau       : amp (constant)
            Ii_tau       : amp (constant)
            Ie_th        : amp (constant)
            Ii_th        : amp (constant)
            kn_syn       : 1 (constant)
            kp_syn       : 1 (constant)
            Ut_syn       : volt (constant)
            Io_syn       : amp (constant)
            Csyn         : farad (constant)
            """,
            'on_pre': """
             Ie_syn += Iw_e*Ie_gain*(weight>0)/(Itau_e*((Ie_gain/Ie_syn)+1))
             Ii_syn += Iw_i*Ii_gain*(weight<0)/(Itau_i*((Ii_gain/Ii_syn)+1))
              """,
            'on_post': """ """,
              }
