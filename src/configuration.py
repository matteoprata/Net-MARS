
import src.constants as co


class Configuration:
    def __init__(self):
        self.seed = 0
        self.graph_name = co.GraphName.MINNESOTA.value
        self.destruction = co.Destruction.COMPLETE.value

        self.destruction_type = co.Destruction.GAUSSIAN
        self.destruction_precision = 100  # density of the [1,0] grid
        self.n_destruction = 3

        # constants
        self.path_to_graph = "data/graphs/"

