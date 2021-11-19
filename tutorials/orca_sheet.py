from orca_wta import ORCA_WTA
from orca_params import excitatory_population_size, ei_ratio


class orca_sheet:
    def __init__(self, network):
        self.network = network
        self.layers = []
        self.interlaminar_connections = []

    def create_column(self):
        num_exc = 49
        layer = 'L23'
        self.layer_23 = ORCA_WTA(num_exc_neurons=num_exc*exc_pop_proportion[layer],
                                 ei_ratio=ei_ratio[layer],
                                 layer=layer)
        self.layers.append(self.layer_23)
        layer = 'L4'
        self.layer_4 = ORCA_WTA(num_exc_neurons=num_exc*exc_pop_proportion[layer],
                                ei_ratio=ei_ratio[layer],
                                layer=layer)
        self.layers.append(self.layer_4)
        layer = 'L5'
        self.layer_5 = ORCA_WTA(num_exc_neurons=num_exc*exc_pop_proportion[layer],
                                ei_ratio=ei_ratio[layer],
                                layer=layer)
        self.layers.append(self.layer_5)
        layer = 'L6'
        self.layer_6 = ORCA_WTA(num_exc_neurons=num_exc*exc_pop_proportion[layer],
                                ei_ratio=ei_ratio[layer],
                                layer=layer)
        self.layers.append(self.layer_6)

        # Connection layers
        self.layer23.add_input(layer_4._groups['pyr_cells'],
                               'L4_L23',
                               ['pyr_cells'],
                               'stdp',
                               'excitatory')

    def add_layers(self):
        for layer in self.layers:
            self.network.add(layer)

    def add_interlaminar_connections(self):
        pass
