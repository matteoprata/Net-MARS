
import src.constants as co


class Configuration:
    def __init__(self):
        self.seed = 1
        self.graph_name = co.GraphName.INTEROUTE.value

        self.destruction_show_plot = False
        self.destruction_save_plot = True

        self.destruction_type = co.Destruction.UNIFORM
        self.destruction_uniform_quantity = .3

        self.destruction_width = .04
        self.destruction_precision = 1000  # density of the [1,0] grid
        self.n_destruction = 5

        self.n_demand_pairs = 2
        self.demand_capacity = 2
        self.supply_capacity = 5

