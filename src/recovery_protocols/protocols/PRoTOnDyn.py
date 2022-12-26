
import src.plotting.graph_plotting as pg
from src.preprocessing.graph_preprocessing import *
from src.preprocessing.graph_monitoring import *
from src.preprocessing.graph_utils import *
import src.constants as co

from src.recovery_protocols.utils import finder_recovery_path as frp, finder_recovery_path_pick as frpp
from src.monitor_placement_protocols import adding_monitors as mon

from src.recovery_protocols.RecoveryProtocol import RecoveryProtocol


class PRoTOnDyn(RecoveryProtocol):

    file_name = "PRoTOnDyn"
    plot_name = "PRoTOn Dyn"

    mode_path_repairing = co.ProtocolRepairingPath.MIN_COST_BOT_CAP
    mode_path_choosing_repair = co.ProtocolPickingPath.MIN_COST_BOT_CAP
    mode_monitoring = co.ProtocolMonitorPlacement.BUDGET
    mode_monitoring_type = co.PriorKnowledge.TOMOGRAPHY

    plot_marker = "D"
    plot_color_curve = 2

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
            G.nodes[n1][co.ElemAttr.IS_MONITOR.value] = True
            G.nodes[n2][co.ElemAttr.IS_MONITOR.value] = True
            monitors_stats |= {n1, n2}

            # does not look defined for only monitors
            monitors_map[n1] |= {(n1, n2)}
            monitors_map[n2] |= {(n1, n2)}

        self.config.monitors_budget_residual -= len(monitors_stats)

        MISSION_DURATION = 500
        # Viviana: con un processo di Poisson decidiamo gli "arrival time" delle distruzioni dinamiche
        rate = 1 / 30  # tempo medio fra due rotture dinamiche
        num_arrivals = MISSION_DURATION   # numero di arrivi totali. Ne mettiamo uno alto per fare esperimenti lunghi a piacimento
        # ma non ci interessano tutti.
        arrival_time = 30

        destroy_times = np.array(self.poisson_process_dynamic_distruction(rate, num_arrivals, arrival_time, self.config))
        destroy_times = destroy_times[destroy_times < (MISSION_DURATION - 15)]
        count_dis = 0

        no_repairs_prev = 0
        no_repairs = 0
        fixed_paths = []

        print("TIMES DES", destroy_times)
        iter_dest = 0
        # start of the protocol
        while MISSION_DURATION - iter_dest > 0:
            # go on if there are demand edges to satisfy, and still is_feasible
            demand_edges_routed_flow_pp = defaultdict(int)  # (d_edge): flow

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
                     "flow": routed_flow,
                     "monitors": monitors_stats,
                     "packet_monitoring": packet_monitor,
                     "demands_sat": demands_sat,
                     "forced_destr": False}

            for ke in demands_sat:
                stats["demands_sat"][ke].append(0)

            # -------------- 0. Destruction --------------
            # Viviana: la dynamic destruction va fatta quando il numero di riparazioni
            # attuale coincide con il corrente arrival time del processo di poisson
            # no_repairs_prev = len(stats_list[iter-1]["node"]) + len(stats_list[iter-1]["edge"])
            if count_dis < len(destroy_times):
                if (no_repairs_prev < no_repairs and no_repairs >= destroy_times[count_dis]) or (no_repairs_prev == no_repairs and iter_dest >= destroy_times[count_dis]):
                    # first condition: repairs have been done, we beat time with number of repairs
                    # second condition: everything works, so we need to beat time with numer of iterations instead
                    destr_nodes, destr_edges = self.dynamic_destruction(G)
                    print("DOING DESTRUCTION!!!", destr_nodes, destr_edges)
                    print("FIXED_STL!!!", fixed_paths)
                    fixed_paths, fixed_paths_to_remove, stats = self.do_revert_routed_path(G, destr_nodes, destr_edges, fixed_paths, stats)
                    routed_flow = stats["flow"]
                    print("REMOVED!!!", fixed_paths_to_remove)
                    count_dis += 1
                    stats["forced_destr"] = True

            # -------------- 1. Monitoring --------------

            monitors, _, candidate_monitors_dem = mon.new_monitoring_add(G, self.config)
            monitors_map = mon.merge_monitor_maps(monitors_map, candidate_monitors_dem)  # F(n) -> [(d1, d2)]
            stats["monitors"] |= monitors
            monitors_stats = stats["monitors"]

            # >>>> PRUNING HERE
            monitoring = pruning_monitoring_dynamic(G,
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
                return stats_list

            stats_packet_monitoring, demand_edges_to_repair, demand_edges_routed_flow, monitoring_paths, demand_edges_routed_flow_pp, pruned_paths = monitoring
            tomography_over_paths(G, elements_val_id, elements_id_val, self.config.UNK_prior, monitoring_paths)
            fixed_paths += pruned_paths

            routed_flow += sum(demand_edges_routed_flow)
            stats["flow"] = routed_flow
            stats["packet_monitoring"] += stats_packet_monitoring
            packet_monitor = stats["packet_monitoring"]

            for ke in demands_sat:  # every demand edge
                is_new_routing = sum(stats["demands_sat"][ke]) == 0 and is_demand_edge_saturated(G, ke[0], ke[1])  # already routed
                flow = self.config.demand_capacity if is_new_routing else 0
                stats["demands_sat"][ke].append(flow)

            demand_edges = get_demand_edges(G, is_check_unsatisfied=True, is_residual=True)
            print("> Residual demand edges", len(demand_edges), demand_edges)

            # -------------- 2. Decision recovery --------------
            no_repairs_prev = no_repairs
            no_rep_no_cum = 0
            if len(demand_edges) > 0:
                paths_proposed = frp.find_paths_to_repair(self.config.repairing_mode, G, demand_edges_to_repair, get_supply_max_capacity(self.config), is_oracle=self.config.is_oracle_baseline)
                path_to_fix = frpp.find_path_picker(self.config.picking_mode, G, paths_proposed, self.config.repairing_mode, self.config,
                                                    is_oracle=self.config.is_oracle_baseline)

                # assert path_to_fix is not None
                self.do_increase_resistance(G, path_to_fix, self.config)

                d1, d2 = last_repaired_demand = make_existing_edge(path_to_fix[0], path_to_fix[-1])
                self.update_monitor_maps(d1, d2, monitors_non_connections, monitors_connections)

                fixed_nodes, fixed_edges = do_fix_path(G, path_to_fix)
                stats["node"] += fixed_nodes
                stats["edge"] += fixed_edges

                no_repairs += len(fixed_edges) + len(fixed_nodes)
                no_rep_no_cum = len(fixed_edges) + len(fixed_nodes)
            iter_dest += no_rep_no_cum if no_rep_no_cum > 0 else 1
            stats_list.append(stats)
            print(stats)

        return stats_list

    @staticmethod
    def update_monitor_maps(d1, d2, monitors_non_connections, monitors_connections):
        # monitors are not connected PHASE 0
        monitors_non_connections[d1] |= {d2}
        monitors_non_connections[d2] |= {d1}

        monitors_connections[d1] -= {d2}
        monitors_connections[d2] -= {d1}

    @staticmethod
    def cancel_demand_edge(G, path_to_fix):
        print("Path with capacity 0, happened", path_to_fix)
        dd1, dd2 = make_existing_edge(path_to_fix[0], path_to_fix[-1])
        G.edges[dd1, dd2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value] = 0

    @staticmethod
    def dynamic_destruction(G):
        N_EPICENTERS = 3
        HOPS = 3

        destr_nodes, destr_edges = set(), set()
        for i in range(N_EPICENTERS):
            id_epi = np.random.randint(len(G.nodes), size=1)
            id_node_epi = id_epi[0]  # G.nodes[id_epi[0]]
            destr_nodes_temp, destr_edges_temp = k_hop_destruction(G, id_node_epi, HOPS)
            destr_nodes |= destr_nodes_temp
            destr_edges |= destr_edges_temp
        return destr_nodes, destr_edges
        # WHEN DESTROY RESET CAPACITIES Viviana: and stop routing flow of demands traversing the newly broken elements

    @staticmethod
    def poisson_process_dynamic_distruction(rate, num_arrivals, arrival_time, config):
        # Viviana: tutta questa funzione (requires math and random)
        CONST_SEED = 1
        util.set_seed(CONST_SEED)

        poi_process = []
        co = 0
        for i in range(num_arrivals):
            # Get the next probability value from Uniform(0,1)
            p = random.random()

            # Plug it into the inverse of the CDF of Exponential(_lamnbda)
            inter_arrival_time = -math.log(1.0 - p) / rate

            # Add the inter-arrival time to the running sum
            arrival_time = arrival_time + inter_arrival_time

            # print it all out
            # print(str(p) + ',' + str(inter_arrival_time) + ',' + str(arrival_time))
            cat = np.ceil(arrival_time)

            if co == 0 or (co > 0 and poi_process[co-1] != cat):
                poi_process += [cat]
                co += 1

        util.set_seed(config.seed)
        return poi_process

    @staticmethod
    def do_increase_resistance(G, path, config):
        """ increase the resistance of all the components in a path """
        n0 = path[0]
        G.nodes[n0][co.ElemAttr.RESISTANCE_TO_DESTRUCTION.value] = config.uniform_resistance_destruction_reset
        for i in range(len(path)-1):
            n1, n2 = make_existing_edge(path[i], path[i + 1])
            G.nodes[n2][co.ElemAttr.RESISTANCE_TO_DESTRUCTION.value] = config.uniform_resistance_destruction_reset
            G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESISTANCE_TO_DESTRUCTION.value] = config.uniform_resistance_destruction_reset

    @staticmethod
    def do_revert_routed_path(G, destr_nodes, destr_edges, fixed_paths, stats_iter):
        """ for paths that are broken, restore capacities """
        fixed_paths_to_remove = []
        for pa, pru in fixed_paths:
            is_path_to_revert = False
            for i in range(len(pa) - 1):
                n1, n2 = make_existing_edge(pa[i], pa[i + 1])
                is_path_to_revert = n1 in destr_nodes or n2 in destr_nodes or (n1, n2) in destr_edges

            if is_path_to_revert:
                ns, nd = make_existing_edge(pa[0], pa[-1])
                G.edges[ns, nd, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value] += pru

                for i in range(len(pa) - 1):
                    n1, n2 = make_existing_edge(pa[i], pa[i + 1])
                    G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value] = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.CAPACITY.value]
                fixed_paths_to_remove.append((pa, pru))
                stats_iter["flow"] -= pru
                stats_iter["demands_sat"][(ns, nd)][-1] -= pru

        for el in fixed_paths_to_remove:  # broken paths
            fixed_paths.remove(el)
        return fixed_paths, fixed_paths_to_remove, stats_iter