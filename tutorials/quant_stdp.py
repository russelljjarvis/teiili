from brian2.units import *

quant_stdp = {
    'model':'''
        w_plast : 1
        dApre/dt = int(Apre * decay_stdp_Apre + decay_probability_Apre)/second : 1 (clock-driven)
        dApost/dt = int(Apost * decay_stdp_Apost + decay_probability_Apost)/second : 1 (clock-driven)
        decay_probability_Apre = rand() : 1 (constant over dt)
        decay_probability_Apost = rand() : 1 (constant over dt)

        decay_stdp_Apre = taupre/(taupre + dt) : 1
        decay_stdp_Apost = taupost/(taupost + dt) : 1

        taupre : second (constant)
        taupost : second (constant)
        w_max : 1
        lr : 1 (shared)
        deltaApre : 1 (constant)
        deltaApost : 1 (constant)

        rand_pre1 : 1
        rand_pre2 : 1
        rand_post1 : 1
        rand_post2 : 1
        stdp_thres : 1 (constant)
        rand_num_bits_pre1 : 1 # Number of bits of random number generated for Apre
        rand_num_bits_post1 : 1 # Number of bits of random number generated for Apost
        ''',
    'on_pre':'''
        ge_post += w_plast
        Apre += deltaApre
        Apre = clip(Apre, 0, 15)
        rand_pre1 = ceil(rand() * (2**rand_num_bits_pre1-1))
        rand_pre2 = ceil(rand() * (2**rand_num_bits_pre1-1))
        w_plast = clip(w_plast - lr*int(lastspike_post!=lastspike_pre)*int(rand_pre1 < Apost)*int(rand_pre2 <= stdp_thres), 0, w_max)
        ''',
    'on_post':'''
        Apost += deltaApost
        Apost = clip(Apost, 0, 15)
        rand_post1 = ceil(rand() * (2**rand_num_bits_post1-1))
        rand_post2 = ceil(rand() * (2**rand_num_bits_post1-1))
        w_plast = clip(w_plast + lr*int(lastspike_post!=lastspike_pre)*int(rand_post1 < Apre)*int(rand_post2 <= stdp_thres), 0, w_max)
        ''',
    'parameters':{
        'taupre': '20*msecond',
        'taupost': '20*msecond',
        'w_max': '.01',
        'lr': '.0001',
        'deltaApre': '15',
        'deltaApost': '15',
        'rand_num_bits_pre1': '4',
        'rand_num_bits_post1': '4',
        'stdp_thres': 1
        }
    }

