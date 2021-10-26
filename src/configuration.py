
import src.constants as co


class Configuration:
    def __init__(self):
        self.seed = 0
        self.graph_name = co.GraphName.SINET.value
        self.destruction = co.Destruction.COMPLETE.value

        self.destruction_show_plot = False
        self.destruction_save_plot = True

        self.destruction_type = co.Destruction.GAUSSIAN
        self.destruction_width = .05
        self.destruction_precision = 1000  # density of the [1,0] grid
        self.n_destruction = 3

        self.n_demand_pairs = 10
        self.demand_capacity = 10

        self.supply_capacity = 1

