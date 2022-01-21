from orca_wta import orcaWTA
from orca_params import conn_desc, pop_desc


class orcaColumn:
    def __init__(self):
        self.layers = []
        self.interlaminar_connections = []

    # TODO col_desc?
    # TODO get id (layer) and Nsize from param dict? Stack layers?
    def create_column(self, pop_params=):
        temp_wtas  = {}
        for wta_id, params in pop_params.groups.items():
            pass

        num_exc = 49
        layer = 'L23'
        self.layer_23 = orcaWTA(num_exc_neurons=num_exc*exc_pop_proportion[layer],
                                 ei_ratio=ei_ratio[layer],
                                 layer=layer)
        self.layers.append(self.layer_23)
        layer = 'L4'
        self.layer_4 = orcaWTA(num_exc_neurons=num_exc*exc_pop_proportion[layer],
                                ei_ratio=ei_ratio[layer],
                                layer=layer)
        self.layers.append(self.layer_4)
        layer = 'L5'
        self.layer_5 = orcaWTA(num_exc_neurons=num_exc*exc_pop_proportion[layer],
                                ei_ratio=ei_ratio[layer],
                                layer=layer)
        self.layers.append(self.layer_5)
        layer = 'L6'
        self.layer_6 = orcaWTA(num_exc_neurons=num_exc*exc_pop_proportion[layer],
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
