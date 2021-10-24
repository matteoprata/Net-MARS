
from src.constants import *


class Configuration:
    def __init__(self):
        self.seed = 0
        self.graph_name = GraphName.MINNESOTA.value
        self.destruction = Destruction.COMPLETE.value

        self.destruction_precision = 100  # density of the [1,0] grid
        self.n_destruction = 3

        # constants
        self.path_to_graph = "data/graphs/"

