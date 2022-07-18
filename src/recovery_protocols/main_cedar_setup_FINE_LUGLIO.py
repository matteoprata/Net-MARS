
import src.plotting.graph_plotting as pg
from src.preprocessing.graph_preprocessing import *
from src.preprocessing.graph_monitoring import *
from src.preprocessing.graph_utils import *
import src.constants as co

from src.recovery_protocols import finder_recovery_path as frp
from src.recovery_protocols import finder_recovery_path_pick as frpp
from src.monitor_placement_protocols import adding_monitors as mon

import time


def run(config):
    stats_list = []

    # read graph and print stats
    G, elements_val_id, elements_id_val = init_graph(co.PATH_TO_GRAPH, config.graph_path, config.supply_capacity, config)
    print_graph_info(G)

    # normalize coordinates and break components
    dim_ratio = scale_coordinates(G)

    distribution, broken_nodes, broken_edges, perc_broken_elements = destroy(G, config.destruction_type, config.destruction_precision, dim_ratio,
                                                                             config.destruction_width, config.n_destruction, config.graph_dataset, config.seed, ratio=config.destruction_quantity,
                                                                             config=config)

    # add_demand_endpoints
    if config.is_demand_clique:
        add_demand_clique(G, config.n_demand_clique, config.demand_capacity, config)
    else:
        add_demand_pairs(G, config.n_demand_pairs, config.demand_capacity, config)

    # path = "data/porting/graph-s|{}-g|{}-np|{}-dc|{}-pbro|{}-supc|{}.json".format(config.seed, config.graph_dataset.name, config.n_demand_clique,
    #                                                                                    config.demand_capacity, config.destruction_quantity,
    #                                                                                    config.supply_capacity[0])
    # util.save_porting_dictionary(G, path)
    # util.enable_print()

    # feasible = is_feasible(G, is_fake_fixed=True)
    # util.enable_print()
    # if not feasible:
    #     print("WARNING! No feasible")
    # return

    pg.plot(G, config.graph_path, distribution, config.destruction_precision, dim_ratio,
            config.destruction_show_plot, config.destruction_save_plot, config.seed, "TRU", co.PlotType.TRU, config.destruction_quantity)

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

    # if config.monitoring_type == co.PriorKnowledge.FULL:
    #     gain_knowledge_all(G)

    # assert config.monitors_budget == -1 or config.monitors_budget >= len(get_demand_nodes(G)), \
    #     "budget is {}, demand nodes are {}".format(config.monitors_budget, len(get_demand_nodes(G)))

    if config.monitors_budget == -1:  # -1 budget means to set automatically as get_demand_nodes(G)
        config.monitors_budget = get_demand_nodes(G)

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

        do_k_monitoring(G, n1, config.k_hop_monitoring)
        do_k_monitoring(G, n2, config.k_hop_monitoring)

    config.monitors_budget_residual -= len(monitors_stats)  # TODO CHECK

    # start of the protocol
    while len(get_demand_edges(G, is_check_unsatisfied=True)) > 0:
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

            d_edge = make_existing_edge(G, d1, d2)
            demand_edges_routed_flow_pp[d_edge] += quantity_pruning
            stats["flow"] = routed_flow

        # -------------- 0. Monitor placement --------------
        # # # TODO: REMEMBER SHIT
        # _, _, candidate_monitors_dem = mon.new_monitoring_add(G, config)
        # monitors_map = mon.merge_monitor_maps(monitors_map, candidate_monitors_dem)  # F(n) -> [(d1, d2)]
        #
        # # >>>> PRUNING HERE
        # monitoring = pruning_monitoring_dummy(G,
        #                                       stats["packet_monitoring"],
        #                                       config.monitoring_messages_budget,
        #                                       monitors_map,
        #                                       monitors_connections,
        #                                       monitors_non_connections,
        #                                       last_repaired_demand,
        #                                       config)
        #
        # stats_packet_monitoring, demand_edges_to_repair, demand_edges_routed_flow, monitoring_paths = monitoring
        # tomography_over_paths(G, elements_val_id, elements_id_val, config.UNK_prior, monitoring_paths)
        # make_components_known(G)

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
            path_to_fix = frpp.find_path_picker(co.ProtocolPickingPath.CEDAR_LIKE_MIN, G, PK, None, False)
            print("Chose to repair", path_to_fix)
            fixed_nodes, fixed_edges = do_fix_path(G, path_to_fix)
            stats["edge"] += fixed_nodes
            stats["node"] += fixed_edges

            d1, d2 = last_repaired_demand = make_existing_edge(G, path_to_fix[0], path_to_fix[-1])
            # update_monitor_maps(d1, d2, monitors_non_connections, monitors_connections)

        else:  # Pk non ha dei path
            if config.monitors_budget_residual > 0:
                v = best_centrality_node(G)
                config.monitors_budget_residual -= 1
                fixed_node = do_repair_node(G, v)
                if fixed_node:
                    stats["node"] += [fixed_node]

                monitors_stats |= {v}
                G.nodes[v][co.ElemAttr.IS_MONITOR.value] = True
                stats["monitors"] |= monitors_stats

                # k-discovery
                packets_monitoring = do_k_monitoring(G, v, config.k_hop_monitoring)
                stats["packet_monitoring"] = packets_monitoring
            else:
                force_state = co.NodeState.WORKING
                make_components_known_to_state(G, force_state.value)
                print("No monitors left. All nodes are set to {}.".format(force_state.name))
                # stats_list.append(stats)
                # demand_log(demands_sat, demand_edges_routed_flow_pp, stats)
                # return stats_list

        demand_log(demands_sat, demand_edges_routed_flow_pp, stats)
        stats_list.append(stats)
    return stats_list


def update_monitor_maps(d1, d2, monitors_non_connections, monitors_connections):
    # monitors are not connected PHASE 0
    monitors_non_connections[d1] |= {d2}
    monitors_non_connections[d2] |= {d1}

    monitors_connections[d1] -= {d2}
    monitors_connections[d2] -= {d1}


def cancel_demand_edge(G, path_to_fix):
    print("Path with capacity 0, happened", path_to_fix)
    dd1, dd2 = make_existing_edge(G, path_to_fix[0], path_to_fix[-1])
    G.edges[dd1, dd2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value] = 0


def demand_log(demands_sat, demand_edges_routed_flow_pp, stats):
    for ke in demands_sat:
        flow = demand_edges_routed_flow_pp[ke] if ke in demand_edges_routed_flow_pp.keys() else 0
        stats["demands_sat"][ke].append(flow)
