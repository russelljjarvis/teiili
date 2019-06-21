from brian2.units import *
DPIstdp_gm = {'model':
'''
        dIe_syn/dt = (-Ie_syn - Ie_gain + 2*Io_syn*(Ie_syn<=Io_syn))/(tausyne*((Ie_gain/Ie_syn)+1)) : amp (clock-driven)
        dIi_syn/dt = (-Ii_syn - Ii_gain + 2*Io_syn*(Ii_syn<=Io_syn))/(tausyni*((Ii_gain/Ii_syn)+1)) : amp (clock-driven)

        Ie{input_number}_post = Ie_syn : amp (summed)
        Ii{input_number}_post = -Ii_syn : amp (summed)

        weight : 1
        w_plast : 1

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

      dApre/dt = -Apre / taupre : 1 (event-driven)
      dApost/dt = -Apost / taupost : 1 (event-driven)
      w_max: 1 (constant)
      taupre : second (constant)
      taupost : second (constant)
      dApre : 1 (constant)
      Q_diffAPrePost : 1 (constant)
      ''',
'on_pre':
'''

        Ie_syn += Iw_e*w_plast*Ie_gain*Ipred_pre*(weight>0)/(Itau_e*((Ie_gain/Ie_syn)+1))
        Ii_syn += Iw_i*w_plast*Ii_gain*Ipred_pre*(weight<0)/(Itau_i*((Ii_gain/Ii_syn)+1))

      Apre += dApre*w_max
      w_plast = clip(w_plast + Apost, 0, w_max) ''',
'on_post':
'''

      Apost += -dApre * (taupre / taupost) * Q_diffAPrePost * w_max
      w_plast = clip(w_plast + Apre, 0, w_max)
''',
'parameters':
{
'Csyn' : '1.5 * pfarad',
'Io_syn' : '0.5 * pamp',
'Ie_th' : '10. * pamp',
'taupre' : '10. * msecond',
'Ut_syn' : '25. * mvolt',
'Ii_syn' : '0.5 * pamp',
'Ie_tau' : '10. * pamp',
'taupost' : '10. * msecond',
'Q_diffAPrePost' : '1.05',
'baseweight_i' : '7. * pamp',
'Ii_tau' : '10. * pamp',
'baseweight_e' : '7. * pamp',
'w_plast' : '0',
'Ie_syn' : '0.5 * pamp',
'kn_syn' : '0.75',
'kp_syn' : '0.66',
'Ii_th' : '10. * pamp',
'w_max' : '1.0',
'dApre' : '0.01',
}
}