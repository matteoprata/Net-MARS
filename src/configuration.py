import numpy as np

import src.constants as co


class Configuration:
    def __init__(self):
        self.seed = 10

        self.mute_log = False
        self.algo_name = co.AlgoName.CEDARNEW.value
        self.graph_dataset = co.GraphName.MINNESOTA
        self.graph_path = self.graph_dataset.value

        self.destruction_show_plot = False
        self.destruction_save_plot = False

        self.destruction_type = co.Destruction.UNIFORM
        self.destruction_uniform_quantity = .3

        self.destruction_width = .05
        self.destruction_precision = 1000  # density of the [1,0] grid
        self.n_destruction = 2

        self.demand_capacity: float = 10.0  # if this > that, multiple paths required to fix
        self.supply_capacity = (150, 200)  # (50, 71)

        # Clique world
        self.is_demand_clique = True
        self.n_demand_clique = 8

        # Edges world
        self.n_demand_pairs = 5

        self.rand_generator_capacities = None
        self.monitoring_type = co.PriorKnowledge.TOMOGRAPHY

        self.monitors_budget = 22  # int(self.n_demand_clique + (self.n_demand_clique * (self.n_demand_clique-1) / 2) / 2)  # n + ( n*(n-1)/2 ) / 4
        self.monitoring_messages_budget = np.inf
