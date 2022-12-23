import numpy as np
import multiprocessing
import src.constants as co


class Configuration:
    def __init__(self):
        self.seed = None

        self.is_log = False
        self.algo_name = None
        self.graph_dataset = co.GraphName.MINNESOTA
        self.graph_path = self.graph_dataset.value

        self.destruction_show_plot = False
        self.destruction_save_plot = False

        self.destruction_type = co.Destruction.GAUSSIAN_PROGRESSIVE
        self.destruction_quantity = None

        self.destruction_width = .05
        self.destruction_precision = 1000  # density of the [1,0] grid
        self.n_destruction = 2

        self.demand_capacity: float = 10.0  # if this > that, multiple paths required to fix
        self.supply_capacity = (80, 81)

        # clique world
        self.is_demand_clique = True
        self.demand_clique_factor = 0.5    # unused
        self.n_nodes_demand_clique = None  # number of nodes of the clique CONSTANT (July)

        # Edges world
        self.n_edges_demand = None

        self.rand_generator_capacities = None
        self.rand_generator_path_choice = None
        self.monitoring_type = co.PriorKnowledge.TOMOGRAPHY

        self.monitors_budget = 25
        self.monitors_budget_residual = None
        self.monitoring_messages_budget = np.inf

        self.is_minnesota_backbone_on = True
        self.backbone_capacity = 160  # 320
        # self.n_backbone_pairs = 0
        self.percentage_flow_backbone = 1  # increase in flow quantity

        self.repairing_mode = None  # co.ProtocolRepairingPath.MIN_COST_BOT_CAP
        self.picking_mode = None  # co.ProtocolPickingPath.MIN_COST_BOT_CAP

        # self.is_adaptive_prior = True
        self.is_oracle_baseline = False  # baseline TOMOCEDAR
        self.fixed_unvarying_seed = 10
        self.experiment_ind_var = None
        self.edges_list_var = None  # a dictionary containing the demand edges in the previous runs if the demand pairs vary
        self.edges_list_path = "data/demand_edges/edges_list"

        self.is_dynamic_prior = True
        self.UNK_prior = None

        self.protocol_monitor_placement = None  # co.ProtocolMonitorPlacement.STEP_BY_STEP
        self.is_exhaustive_paths = False

        self.force_recompute = True
        self.log_execution_details = True

        self.is_parallel = False

        self.k_hop_monitoring = 1
        self.is_IP_routing = False  # weather to do IP, or self routing
        self.uniform_resistance_destruction_init = 0.2    # at the beginning
        self.uniform_resistance_destruction_reset = 0.97   # at reset
        self.fix_with_seed_mode = False  # fix demand edges or epicenters
        self.is_cluster_execution = multiprocessing.cpu_count() > 10

    def n_edges_given_n_nodes(self, n_nodes):
        """ number of edges given pruned clique """
        return int(n_nodes*(n_nodes-1)/2 * self.demand_clique_factor)
