import src.plotting.graph_plotting as pg
from src.preprocessing.graph_preprocessing import *
from src.preprocessing.graph_monitoring import *
from src.preprocessing.graph_utils import *
import src.constants as co

from src.recovery_protocols.utils import finder_recovery_path_pick as frpp
from src.recovery_protocols.RecoveryProtocol import RecoveryProtocol


class CeDAR(RecoveryProtocol):
    file_name = "CeDAR"
    plot_name = "CeDAR"

    plot_marker = "s"
    plot_color_curve = 3

    def __init__(self, config):
        super().__init__(config)

    def run(self):
        stats_list = []

        # read graph and print stats
        G, elements_val_id, elements_id_val = init_graph(co.PATH_TO_GRAPH, self.config.graph_path, self.config.supply_capacity, self.config)
        print_graph_info(G)

        # normalize coordinates and break components
        dim_ratio = scale_coordinates(G)

        distribution, broken_nodes, broken_edges, perc_broken_elements = destroy(G, self.config.destruction_type, self.config.destruction_precision, dim_ratio,
                                                                                 self.config.destruction_width, self.config.n_destruction, self.config.graph_dataset, self.config.seed, ratio=self.config.destruction_quantity,
                                                                                 config=self.config)

        # add_demand_endpoints
        if self.config.is_demand_clique:
            add_demand_clique(G, self.config)
        else:
            add_demand_pairs(G, self.config.n_edges_demand, self.config.demand_capacity, self.config)

        pg.plot(G, self.config.graph_path, distribution, self.config.destruction_precision, dim_ratio,
                self.config.destruction_show_plot, self.config.destruction_save_plot, self.config.seed, "TRU", co.PlotType.TRU, self.config.destruction_quantity)

        # hypothetical routability
        if not is_feasible(G, is_fake_fixed=True):
            print("This instance is not solvable. Check the number of demand edges, theirs and supply links capacity.\n\n\n")
            return

        # repair demand edges
        demand_node = get_demand_nodes(G)
        for dn in demand_node:
            do_repair_node(G, dn)
            # INITIAL NODES repairs are not counted in the stats

        iter = 0
        routed_flow = 0
        packet_monitor = 0
        monitors_stats = set()
        demands_sat = {d: [] for d in get_demand_edges(G, is_capacity=False)}  # d1: [0, 1, 1, 0, 10] // demands_sat[d].append(0)

        # set as monitors all the nodes that are demand endpoints
        monitors_map = defaultdict(set)
        monitors_connections = defaultdict(set)
        monitors_non_connections = defaultdict(set)

        last_repaired_demand = None

        # ADD preliminary monitors
        for n1, n2, _ in get_demand_edges(G):
            do_repair_node(G, n1)
            do_repair_node(G, n2)

            G.nodes[n1][co.ElemAttr.IS_MONITOR.value] = True
            G.nodes[n2][co.ElemAttr.IS_MONITOR.value] = True
            monitors_stats |= {n1, n2}

            # does not look defined for only monitors
            monitors_map[n1] |= {(n1, n2)}
            monitors_map[n2] |= {(n1, n2)}

            packet_monitor += do_k_monitoring(G, n1, self.config.k_hop_monitoring)
            packet_monitor += do_k_monitoring(G, n2, self.config.k_hop_monitoring)

        self.config.monitors_budget_residual -= len(monitors_stats)

        # start of the protocol
        while len(get_demand_edges(G, is_check_unsatisfied=True)) > 0:
            # go on if there are demand edges to satisfy, and still is_feasible
            print("\n\n", "#" * 40, "BEGIN ITERATION", "#" * 40)

            # check if the graph is still routbale on tot graph,
            if not is_feasible(G, is_fake_fixed=True):
                print("This instance is no more routable!")
                return stats_list

            iter += 1
            print("ITER", iter)

            # packet_monitor -- monitors paced up to iteration i
            # monitors -- monitors placed up to now (no duplicates)
            stats = {"iter": iter,
                     "node": [],
                     "edge": [],
                     "flow": 0,
                     "monitors": monitors_stats,
                     "packet_monitoring": packet_monitor,
                     "demands_sat": demands_sat}

            # PRUNING:
            SG = get_supply_graph(G)
            if last_repaired_demand is not None:
                d1, d2 = last_repaired_demand
                path, _, _, _ = mxv.protocol_routing_IP(SG, d1, d2)

                quantity_pruning = do_prune(G, path)
                routed_flow += quantity_pruning
                print("pruned", quantity_pruning, "on", path)

                d_edge = make_existing_edge(d1, d2)
                stats["flow"] = routed_flow

            self.demand_log(G, demands_sat, stats, self.config)

            SG = get_supply_graph(G)
            paths = []
            for d1, d2, _ in get_demand_edges(G, is_check_unsatisfied=True):
                path, _, _ = mxv.protocol_repair_cedarlike(SG, d1, d2)
                paths.append(path)

            # filter paths
            PK = []  # PK
            for pa in paths:
                if is_known_path(G, pa):
                    PK.append(pa)

            if len(PK) > 0:  # Pk ha dei path
                path_to_fix = frpp.find_path_picker(co.ProtocolPickingPath.CEDAR_LIKE_MIN, G, PK, None, self.config, False)
                print("Chose to repair", path_to_fix)
                fixed_nodes, fixed_edges = do_fix_path(G, path_to_fix)
                stats["edge"] += fixed_nodes
                stats["node"] += fixed_edges

                d1, d2 = last_repaired_demand = make_existing_edge(path_to_fix[0], path_to_fix[-1])
                # update_monitor_maps(d1, d2, monitors_non_connections, monitors_connections)

            else:  # Pk non ha dei path
                if self.config.monitors_budget_residual > 0:
                    v = best_centrality_node(G)
                    self.config.monitors_budget_residual -= 1
                    fixed_node = do_repair_node(G, v)
                    if fixed_node:
                        stats["node"] += [fixed_node]

                    monitors_stats |= {v}
                    G.nodes[v][co.ElemAttr.IS_MONITOR.value] = True
                    stats["monitors"] |= monitors_stats

                    # k-discovery
                    packet_monitor += do_k_monitoring(G, v, self.config.k_hop_monitoring)
                    stats["packet_monitoring"] = packet_monitor
                else:
                    force_state = co.NodeState.WORKING
                    make_components_known_to_state(G, force_state.value)
                    print("No monitors left. All nodes are set to {}.".format(force_state.name))
                    # stats_list.append(stats)
                    # demand_log(demands_sat, demand_edges_routed_flow_pp, stats)
                    # return stats_list
            stats_list.append(stats)
        return stats_list

    @staticmethod
    def demand_log(G, demands_sat, stats, config):
        for ke in demands_sat:  # every demand edge
            is_new_routing = sum(stats["demands_sat"][ke]) == 0 and is_demand_edge_saturated(G, ke[0], ke[1])  # already routed
            flow = config.demand_capacity if is_new_routing else 0
            stats["demands_sat"][ke].append(flow)
