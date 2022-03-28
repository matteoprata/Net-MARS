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

        self.destruction_type = co.Destruction.GAUSSIAN_PROGRESSIVE
        self.destruction_quantity = .2

        self.destruction_width = .05
        self.destruction_precision = 1000  # density of the [1,0] grid
        self.n_destruction = 2

        self.demand_capacity: float = 10.0  # if this > that, multiple paths required to fix
        self.supply_capacity = (150, 200)  # (50, 71)

        # Clique world
        self.is_demand_clique = True
        self.n_demand_clique = 8

        # Edges world
        self.n_demand_pairs = 8

        self.rand_generator_capacities = None
        self.rand_generator_path_choice = None
        self.monitoring_type = co.PriorKnowledge.TOMOGRAPHY

        self.monitors_budget = 10
        self.monitors_budget_residual = self.monitors_budget
        self.monitoring_messages_budget = np.inf

        self.n_backbone_pairs = 5
        self.percentage_flow_backbone = .5  # increase in flow quantity

        self.repairing_mode = None  # co.ProtocolRepairingPath.MIN_COST_BOT_CAP
        self.picking_mode = None  # co.ProtocolPickingPath.MIN_COST_BOT_CAP

        # self.is_adaptive_prior = True
        self.is_oracle_baseline = False  # baseline TOMOCEDAR
        self.is_xindvar_destruction = True   # the X axis destruction varies
        self.fixed_unvarying_seed = 0

        self.is_dynamic_prior = True
        self.UNK_prior = None

        self.protocol_monitor_placement = None  # co.ProtocolMonitorPlacement.STEP_BY_STEP
        self.is_exhaustive_paths = True

        self.force_recompute = False
