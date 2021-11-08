
import src.constants as co


class Configuration:
    def __init__(self):
        self.seed = 15
        self.graph_name = co.GraphName.MINNESOTA.value

        self.destruction_show_plot = False
        self.destruction_save_plot = True

        self.destruction_type = co.Destruction.UNIFORM
        self.destruction_uniform_quantity = .3

        self.destruction_width = .02
        self.destruction_precision = 1000  # density of the [1,0] grid
        self.n_destruction = 2

        self.n_demand_clique = 5
        self.is_demand_clique = False

        self.n_demand_pairs = 15
        self.demand_capacity = 2
        self.supply_capacity = 30

