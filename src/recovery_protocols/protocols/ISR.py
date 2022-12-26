from src.preprocessing.graph_preprocessing import *
from src.preprocessing.graph_monitoring import *
from src.preprocessing.graph_utils import *
import src.constants as co

import src.utilities.util_routing_stpath as mxv

import src.preprocessing.graph_utils as gru
from gurobipy import *

from src.recovery_protocols.RecoveryProtocol import RecoveryProtocol
from abc import ABC


class ISR(RecoveryProtocol, ABC):
    """ Abstract class, can only be inherited. """

    def __init__(self, config):
        super().__init__(config)

    def prepare_run(self):
        stats_list = []

        # read graph and print stats
        G, elements_val_id, elements_id_val = init_graph(co.PATH_TO_GRAPH, self.config.graph_path, self.config.supply_capacity, self.config)
        print_graph_info(G)

        # normalize coordinates and break components
        dim_ratio = scale_coordinates(G)

        distribution, broken_nodes, broken_edges, perc_broken_elements = destroy(G,
                                                                                 self.config.destruction_type,
                                                                                 self.config.destruction_precision, dim_ratio,
                                                                                 self.config.destruction_width,
                                                                                 self.config.n_destruction, self.config.graph_dataset,
                                                                                 self.config.seed,
                                                                                 ratio=self.config.destruction_quantity,
                                                                                 config=self.config)

        # add_demand_endpoints
        if self.config.is_demand_clique:
            add_demand_clique(G, self.config)
        else:
            add_demand_pairs(G, self.config.n_edges_demand, self.config.demand_capacity, self.config)

        # hypothetical routability
        if not is_feasible(G, is_fake_fixed=True):
            print(
                "This instance is not solvable. Check the number of demand edges, theirs and supply links capacity.\n\n\n")
            return

        # repair demand edges
        demand_node = get_demand_nodes(G)
        for dn in demand_node:
            do_repair_node(G, dn)
            # INITIAL NODES repairs are not counted in the stats

        iter = 0
        # true ruotability

        routed_flow = 0
        monitors_stats = set()
        demands_sat = {d: [] for d in
                       get_demand_edges(G, is_capacity=False)}  # d1: [0, 1, 1, 0, 10] // demands_sat[d].append(0)

        packet_monitor = 0
        for n1, n2, _ in get_demand_edges(G):
            G.nodes[n1][co.ElemAttr.IS_MONITOR.value] = True
            G.nodes[n2][co.ElemAttr.IS_MONITOR.value] = True
            monitors_stats |= {n1, n2}
            packet_monitor += do_k_monitoring(G, n1, self.config.k_hop_monitoring)
            packet_monitor += do_k_monitoring(G, n2, self.config.k_hop_monitoring)

        self.config.monitors_budget_residual -= len(monitors_stats)
        return G, stats_list, monitors_stats, packet_monitor, demands_sat, routed_flow, iter

    @staticmethod
    def demand_log(G, demands_sat, stats, config, is_monotonous=True):
        for ke in demands_sat:  # every demand edge
            is_monotonous_ckeck = not is_monotonous or sum(stats["demands_sat"][ke]) == 0
            is_new_routing = is_monotonous_ckeck and is_demand_edge_saturated(G, ke[0], ke[1])  # already routed
            flow = config.demand_capacity if is_new_routing else 0
            stats["demands_sat"][ke].append(flow)
