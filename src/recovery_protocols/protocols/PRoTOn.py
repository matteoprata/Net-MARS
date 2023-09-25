
import src.plotting.graph_plotting as pg
from src.preprocessing.network_init import *
from src.preprocessing.network_monitoring import *
from src.preprocessing.network_utils import *
import src.constants as co
import time

from src.recovery_protocols.utils import finder_recovery_path as frp, finder_recovery_path_pick as frpp, \
    adding_monitors as mon
from src.recovery_protocols.RecoveryProtocol import RecoveryProtocol


class PRoTOn(RecoveryProtocol):

    file_name = "PRoTOn"
    plot_name = "PRoTOn"

    mode_path_repairing = co.ProtocolRepairingPath.MIN_COST_BOT_CAP
    mode_path_choosing_repair = co.ProtocolPickingPath.MIN_COST_BOT_CAP
    mode_monitoring = co.ProtocolMonitorPlacement.BUDGET
    mode_monitoring_type = co.PriorKnowledge.TOMOGRAPHY

    plot_marker = 0
    plot_color_curve = 2

    def __init__(self, config):
        super().__init__(config)

    def run(self):
        comp_time_st = time.time()

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
        # true ruotability

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

        self.config.monitors_budget_residual -= len(monitors_stats)

        print("DEMAND NODES", get_demand_nodes(G))
        print("DEMAND EDGES", get_demand_edges(G))

        # start of the protocol
        while len(get_demand_edges(G, is_check_unsatisfied=True)) > 0:
            # go on if there are demand edges to satisfy, and still is_feasible
            # demand_edges_routed_flow_pp = defaultdict(int)  # (d_edge): flow

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

            # -------------- 0. Monitor placement --------------
            print("Monitor placement")
            if self.config.protocol_monitor_placement == co.ProtocolMonitorPlacement.BUDGET:
                monitors, _, candidate_monitors_dem = mon.new_monitoring_add(G, self.config)
                monitors_map = mon.merge_monitor_maps(monitors_map, candidate_monitors_dem)  # F(n) -> [(d1, d2)]
                stats["monitors"] |= monitors
                monitors_stats = stats["monitors"]

            # -------------- 1. Tomography, Pruning, Probability --------------
            print("Tomography, Pruning, Probability")
            monitoring = pruning_monitoring(G,
                                            stats["packet_monitoring"],
                                            self.config.monitoring_messages_budget,
                                            monitors_map,
                                            monitors_connections,
                                            monitors_non_connections,
                                            last_repaired_demand,
                                            self.config)

            if monitoring is None:
                stats_list.append(stats)
                print(stats)
                self.on_close(comp_time_st, stats_list)
                return stats_list

            stats_packet_monitoring, demand_edges_to_repair, demand_edges_routed_flow, monitoring_paths, \
            demand_edges_routed_flow_pp = monitoring

            # >>>> PROBABILITY HERE
            tomography_over_paths(G, elements_val_id, elements_id_val, self.config.UNK_prior, monitoring_paths)

            routed_flow += sum(demand_edges_routed_flow)
            stats["flow"] = routed_flow

            packet_monitor += stats_packet_monitoring
            stats["packet_monitoring"] = packet_monitor

            print("DEM-SAT", demands_sat)
            for ke in demands_sat:  # every demand edge
                is_new_routing = sum(stats["demands_sat"][ke]) == 0 and is_demand_edge_saturated(G, ke[0], ke[1])  # already routed
                flow = self.config.demand_capacity if is_new_routing else 0
                stats["demands_sat"][ke].append(flow)

            demand_edges = get_demand_edges(G, is_check_unsatisfied=True, is_residual=True)
            print("> Residual demand edges", len(demand_edges), demand_edges)

            if len(demand_edges) > 0:
                print("Repairing")
                # -------------- 2. Repairing --------------
                paths_proposed = frp.find_paths_to_repair(self.config.repairing_mode, G, demand_edges_to_repair, get_supply_max_capacity(self.config), is_oracle=self.config.is_oracle_baseline)
                path_to_fix = frpp.find_path_picker(self.config.picking_mode, G, paths_proposed, self.config.repairing_mode, self.config,
                                                    is_oracle=self.config.is_oracle_baseline)

                print(paths_proposed)
                assert path_to_fix is not None
                d1, d2 = last_repaired_demand = make_existing_edge(path_to_fix[0], path_to_fix[-1])
                self.update_monitor_maps(d1, d2, monitors_non_connections, monitors_connections)

                fixed_nodes, fixed_edges = do_fix_path_smart(G, path_to_fix)
                stats["node"] += fixed_nodes
                stats["edge"] += fixed_edges

            stats_list.append(stats)
            print(stats)
        self.on_close(comp_time_st, stats_list)
        return stats_list

    def on_close(self, comp_time_st, stats_list):
        # append elapsed time to the last stats object
        # measure computation time
        comp_time_et = time.time()
        elapsed_time = comp_time_et - comp_time_st  # elapsed time in seconds
        stats_list[-1]["execution_sec"] = elapsed_time

    @staticmethod
    def update_monitor_maps(d1, d2, monitors_non_connections, monitors_connections):
        # monitors are not connected PHASE 0
        monitors_non_connections[d1] |= {d2}
        monitors_non_connections[d2] |= {d1}

        monitors_connections[d1] -= {d2}
        monitors_connections[d2] -= {d1}
