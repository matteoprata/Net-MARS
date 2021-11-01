
import src.constants as co


class Configuration:
    def __init__(self):
        self.seed = 0
        self.graph_name = co.GraphName.MINNESOTA.value
        self.destruction = co.Destruction.GAUSSIAN.value

        self.destruction_show_plot = False
        self.destruction_save_plot = True

        self.destruction_type = co.Destruction.GAUSSIAN
        self.destruction_uniform_quantity = .1

        self.destruction_width = .04
        self.destruction_precision = 1000  # density of the [1,0] grid
        self.n_destruction = 5

        self.n_demand_pairs = 5
        self.demand_capacity = 2
        self.supply_capacity = 3

