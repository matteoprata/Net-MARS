
import src.constants as co


class Configuration:
    def __init__(self):
        self.seed = 8

        self.algo_name = co.AlgoName.TOMO_CEDAR_FULL.value
        self.graph_dataset = co.GraphName.MINNESOTA
        self.graph_path = self.graph_dataset.value

        self.destruction_show_plot = False
        self.destruction_save_plot = True

        self.destruction_type = co.Destruction.UNIFORM
        self.destruction_uniform_quantity = .3

        self.destruction_width = .02
        self.destruction_precision = 1000  # density of the [1,0] grid
        self.n_destruction = 2

        self.demand_capacity: float = 15.0  # if this > that, multiple paths required to fix
        self.supply_capacity = (30, 31)

        # Clique world
        self.is_demand_clique = False
        self.n_demand_clique = 4

        # Edges world
        self.n_demand_pairs = 5

        self.rand_generator_capacities = None


        self.prior_knowledge = co.PriorKnowledge.TOMOGRAPHY
