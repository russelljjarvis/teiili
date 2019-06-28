"""
@author: matteo


This module provides a hierarchical building block called OCTA.
            -Online Clustering of Temporal Activity-

Attributes:
    
    octa_params (dict): Dictionary of default parameters for wta.

"""

from teili.building_blocks.building_block import BuildingBlock
from teili.building_blocks.octa import Octa
from teili.core.groups import Neurons, Connections
from teili.stimuli.testbench import WTA_Testbench, OCTA_Testbench

from teili.tools.octa_tools.octa_param import wtaParameters, octaParameters


prefs.codegen.target = "numpy"
#Initializa

class Example_hBB (BuildingBlock):
    
    def __init__(self, name,
                 wtaParams = wtaParameters,
                 octaParams = octaParameters,           
                 neuron_eq_builder=octa_neuron,
                 stacked_inp = True,
                noise= True,
                 monitor=True,
                 debug=False):


        self.num_input_neurons = octaParameters['num_input_neurons']
        self.num_neurons = octaParameters['num_neurons']
        block_params = {}
        block_params.update(wtaParams)
        block_params.update(octaParams)

        BuildingBlock.__init__(self, name,
                               neuron_eq_builder,
                               block_params,
                               debug,
                               monitor)


        self.sub_blocks, self._groups, self.monitors, self.standalone_params = gen_octa(name,
                                                              num_input_neurons=self.num_input_neurons,
                                                              num_neurons=self.num_neurons,
                                                               wtaParams = wtaParams,
                                                               octaParams = octaParams,
                                                               neuron_eq_builder= neuron_eq_builder,
                                                               stacked_inp=noise,
                                                                noise  = noise,
                                                              monitor=monitor,
                                                              debug=debug,
                                                             )

#         Creating handles for neuron groups and inputs
        self.comp_wta = self.sub_blocks['compressionWTA']
        self.pred_wta = self.sub_blocks['predictionWTA']

        self.input_groups.update({'comp_n_spike_gen': self.comp_wta._groups['spike_gen']})
        self.output_groups.update({'comp_n_exc': self.comp_wta._groups['n_exc']})

        self.hidden_groups.update({
                'comp_n_inh' : self.comp_wta._groups['n_inh'] ,
                'pred_n_exc' : self.pred_wta._groups['n_exc'],
                'pred_n_inh' :  self.pred_wta._groups['n_inh']})



def gen_hierarchical_octa():

    test_OCTA_layer_1 =  Octa(name='test_OCTA_layer_1', 
                wtaParams = wtaParameters,
                 octaParams = octaParameters,     
                 neuron_eq_builder=octa_neuron,
                 stacked_inp = True,
                noise= True,
                 monitor=True,
                 debug=True)

    test_OCTA_layer_2 =  Octa(name='test_OCTA_layer_2', 
                wtaParams = wtaParameters,
                 octaParams = octaParameters,     
                 neuron_eq_builder=octa_neuron,
                 stacked_inp = False,
                noise= True,
                 monitor=True,
                 debug=True)

  

	 layer_2_input = Connections(test_OCTA_layer_1.sub_blocks['comp_wta']._groups['n_exc'],
                                    test_OCTA_layer_1.sub_blocks['comp_wta']._groups['spike_gen'],
                                    equation_builder=DPIstdp,
                                    method='euler',
                                    name='layer_2_input')

	 standalone_params = {}
	 standalone_params.update(test_OCTA_layer_1.standalone_params)
	 standalone_params.update(test_OCTA_layer_2.standalone_params)
	   sub_blocks = {
    'test_OCTA_layer_1' : test_OCTA_layer_1
    'test_OCTA_layer_2' : test_OCTA_layer_2
    	}



	 return sub_blocks, group, monitors, standalone_params
	


    Net = teiliNetwork()
    Net.add(test_OCTA, test_OCTA_2)





