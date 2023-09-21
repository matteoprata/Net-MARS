
from src.preprocessing.network_init import *
from src.preprocessing.network_monitoring import *
from src.preprocessing.network_utils import *
import src.constants as co
import src.plotting.graph_plotting as pg

import src.preprocessing.network_utils as gru
from gurobipy import *
from src.recovery_protocols.RecoveryProtocol import RecoveryProtocol


class MinTDS(RecoveryProtocol):
    file_name = "MinTDS"
    plot_name = "MinTDS"

    plot_marker = "p"
    plot_color_curve = 9

    def __init__(self, config):
        super().__init__(config)

    @staticmethod
    def run_header(config):
        stats_list = []

        # read graph and print stats
        G, elements_val_id, elements_id_val = init_graph(co.PATH_TO_GRAPH, config.graph_path, config.supply_capacity, config)
        print_graph_info(G)

        # normalize coordinates and break components
        dim_ratio = scale_coordinates(G)

        distribution, broken_nodes, broken_edges, perc_broken_elements = destroy(G, config.destruction_type,
                                                                                 config.destruction_precision,
                                                                                 dim_ratio,
                                                                                 config.destruction_width,
                                                                                 config.n_destruction,
                                                                                 config.graph_dataset,
                                                                                 config.seed,
                                                                                 ratio=config.destruction_quantity,
                                                                                 config=config)

        # add_demand_endpoints
        if config.is_demand_clique:
            add_demand_clique(G, config)
        else:
            add_demand_pairs(G, config.n_edges_demand, config.demand_capacity, config)

        # hypothetical routability
        if not is_feasible(G, is_fake_fixed=True):
            print( "This instance is not solvable. ",
                   "Check the number of demand edges, theirs and supply links capacity, or graph connectivity.\n\n\n")
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
        demands_sat = {d: [] for d in get_demand_edges(G, is_capacity=False)}  # d1: [0, 1, 1, 0, 10] // demands_sat[d].append(0)

        # add monitors
        packet_monitor = 0
        for n1, n2, _ in get_demand_edges(G):
            G.nodes[n1][co.ElemAttr.IS_MONITOR.value] = True
            G.nodes[n2][co.ElemAttr.IS_MONITOR.value] = True
            monitors_stats |= {n1, n2}
            packet_monitor += do_k_monitoring(G, n1, config.k_hop_monitoring)
            packet_monitor += do_k_monitoring(G, n2, config.k_hop_monitoring)

        config.monitors_budget_residual -= len(monitors_stats)
        print("DEMAND EDGES", get_demand_edges(G))
        print("DEMAND NODES", get_demand_nodes(G))

        pg.plot(G, config.graph_path, distribution, config.destruction_precision, dim_ratio,
                config.destruction_show_plot, config.destruction_save_plot, config.seed, "TRU", co.PlotType.TRU,
                config.destruction_quantity)

        return G, stats_list, monitors_stats, packet_monitor, demands_sat, routed_flow, iter

    def run(self):
        self.run_header(self.config)
