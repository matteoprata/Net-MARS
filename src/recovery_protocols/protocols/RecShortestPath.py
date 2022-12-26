
import src.plotting.graph_plotting as pg
from src.preprocessing.graph_preprocessing import *
from src.preprocessing.graph_monitoring import *
from src.preprocessing.graph_utils import *
import src.constants as co

from src.recovery_protocols.utils import finder_recovery_path as frp, finder_recovery_path_pick as frpp
from src.recovery_protocols.RecoveryProtocol import RecoveryProtocol


class RecShortestPath(RecoveryProtocol):

    file_name = "STPATH"
    plot_name = "STP"

    mode_path_repairing = co.ProtocolRepairingPath.SHORTEST_MINUS
    mode_path_choosing_repair = co.ProtocolPickingPath.RANDOM
    mode_monitoring = co.ProtocolMonitorPlacement.NONE
    mode_monitoring_type = co.PriorKnowledge.DUNNY_IP

    plot_marker = "v"
    plot_color_curve = 0

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

        # path = "data/porting/graph-s|{}-g|{}-np|{}-dc|{}-pbro|{}-supc|{}.json".format(self.config.seed, self.config.graph_dataset.name, self.config.n_demand_clique,
        #                                                                                    self.config.demand_capacity, self.config.destruction_quantity,
        #                                                                                    self.config.supply_capacity[0])
        # util.save_porting_dictionary(G, path)
        # util.enable_print()

        # feasible = is_feasible(G, is_fake_fixed=True)
        # util.enable_print()
        # if not feasible:
        #     print("WARNING! No feasible")
        # return

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
            # -------------- 1. Tomography, Pruning, Probability --------------
            monitoring = dummy_pruning(G)
            stats_packet_monitoring, demand_edges_to_repair, demand_edges_routed_flow, demand_edges_routed_flow_pp = monitoring

            routed_flow += sum(demand_edges_routed_flow)
            stats["flow"] = routed_flow

            packet_monitor += stats_packet_monitoring
            stats["packet_monitoring"] = packet_monitor

            for ke in demands_sat:  # every demand edge
                is_new_routing = sum(stats["demands_sat"][ke]) == 0 and is_demand_edge_saturated(G, ke[0], ke[1])  # already routed
                flow = self.config.demand_capacity if is_new_routing else 0
                stats["demands_sat"][ke].append(flow)

            demand_edges = get_demand_edges(G, is_check_unsatisfied=True, is_residual=True)
            print("> Residual demand edges", len(demand_edges), demand_edges)

            if len(demand_edges) > 0:

                # -------------- 2. Repairing --------------
                paths_proposed = frp.find_paths_to_repair(self.config.repairing_mode, G, demand_edges_to_repair, get_supply_max_capacity(self.config), is_oracle=self.config.is_oracle_baseline)
                path_to_fix = frpp.find_path_picker(self.config.picking_mode, G, paths_proposed, self.config.repairing_mode, self.config,
                                                    is_oracle=self.config.is_oracle_baseline)

                if path_to_fix is None:
                    stats_list.append(stats)
                    print(stats)
                    return stats_list

                # if the protocol SHORTEST_MINUS proposes a 0 capacity edge
                if get_path_residual_capacity(G, path_to_fix) == 0:
                    self.cancel_demand_edge(G, path_to_fix)

                fixed_nodes, fixed_edges = do_fix_path(G, path_to_fix)
                stats["node"] += fixed_nodes
                stats["edge"] += fixed_edges

            stats_list.append(stats)
            print(stats)
        return stats_list

    @staticmethod
    def cancel_demand_edge(G, path_to_fix):
        print("Path with capacity 0, happened", path_to_fix)
        dd1, dd2 = make_existing_edge(path_to_fix[0], path_to_fix[-1])
        G.edges[dd1, dd2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value] = 0

