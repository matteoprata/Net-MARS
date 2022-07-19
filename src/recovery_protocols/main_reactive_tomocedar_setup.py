
import src.plotting.graph_plotting as pg
from src.preprocessing.graph_preprocessing import *
from src.preprocessing.graph_monitoring import *
from src.preprocessing.graph_utils import *
import src.constants as co

from src.recovery_protocols import finder_recovery_path as frp
from src.recovery_protocols import finder_recovery_path_pick as frpp
from src.monitor_placement_protocols import adding_monitors as mon
import src.preprocessing.graph_distruction as dis

import time


def run(config):
    dynamic_destruction(G, config)
    # launch(config)


def launch(config):
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
    if config.protocol_monitor_placement != co.ProtocolMonitorPlacement.NONE:

        for n1, n2, _ in get_demand_edges(G):
            G.nodes[n1][co.ElemAttr.IS_MONITOR.value] = True
            G.nodes[n2][co.ElemAttr.IS_MONITOR.value] = True
            monitors_stats |= {n1, n2}

            # does not look defined for only monitors
            monitors_map[n1] |= {(n1, n2)}
            monitors_map[n2] |= {(n1, n2)}

        config.monitors_budget_residual -= len(monitors_stats)

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

        # -------------- 0. Destruction --------------
        dynamic_destruction()

        # -------------- 1. Monitoring --------------

        monitors, _, candidate_monitors_dem = mon.new_monitoring_add(G, config)
        monitors_map = mon.merge_monitor_maps(monitors_map, candidate_monitors_dem)  # F(n) -> [(d1, d2)]
        stats["monitors"] |= monitors
        monitors_stats = stats["monitors"]

        # >>>> PRUNING HERE
        monitoring = pruning_monitoring_dynamic(G,
                                                stats["packet_monitoring"],
                                                config.monitoring_messages_budget,
                                                monitors_map,
                                                monitors_connections,
                                                monitors_non_connections,
                                                last_repaired_demand,
                                                config)

        stats_packet_monitoring, demand_edges_to_repair, demand_edges_routed_flow, monitoring_paths, demand_edges_routed_flow_pp = monitoring
        tomography_over_paths(G, elements_val_id, elements_id_val, config.UNK_prior, monitoring_paths)

        routed_flow += sum(demand_edges_routed_flow)
        stats["flow"] = routed_flow
        stats["packet_monitoring"] += stats_packet_monitoring
        packet_monitor = stats["packet_monitoring"]

        for ke in demands_sat:
            flow = demand_edges_routed_flow_pp[ke] if ke in demand_edges_routed_flow_pp.keys() else 0
            stats["demands_sat"][ke].append(flow)

        demand_edges = get_demand_edges(G, is_check_unsatisfied=True, is_residual=True)
        print("> Residual demand edges", len(demand_edges), demand_edges)

        # -------------- 2. Decision recovery --------------

        if len(demand_edges) > 0:

            paths_proposed = frp.find_paths_to_repair(config.repairing_mode, G, demand_edges_to_repair, get_supply_max_capacity(config), is_oracle=config.is_oracle_baseline)
            path_to_fix = frpp.find_path_picker(config.picking_mode, G, paths_proposed, config.repairing_mode, is_oracle=config.is_oracle_baseline)

            print(paths_proposed)
            assert path_to_fix is not None
            d1, d2 = last_repaired_demand = make_existing_edge(G, path_to_fix[0], path_to_fix[-1])
            update_monitor_maps(d1, d2, monitors_non_connections, monitors_connections)

            if co.ProtocolRepairingPath.SHORTEST_MINUS:
                if get_path_residual_capacity(G, path_to_fix) == 0:
                    cancel_demand_edge(G, path_to_fix)  # if the protocol SHORTEST_MINUS proposes a 0 capacity edge

                fixed_nodes, fixed_edges = do_fix_path(G, path_to_fix)
                stats["node"] += fixed_nodes
                stats["edge"] += fixed_edges

        stats_list.append(stats)
        print(stats)

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


def dynamic_destruction(G, config, dim_ratio):
    N_EPICENTERS = 1
    HOPS = None

    for i in range(N_EPICENTERS):
        id_epi = np.random.randint(len(G.nodes), size=1)
        id_node_epi = G.nodes[id_epi]
        k_hop_destruction(G, id_node_epi, HOPS)

    # WHEN DESTROY RESET CAPACITIES
    # CHECK MONITORING
    # WHEN REPAIR INCREASE RESISTANCE
