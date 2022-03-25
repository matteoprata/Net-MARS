
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

    # path = "data/porting/graph-s|{}-g|{}-np|{}-dc|{}-uni-pbro|{}.json".format(config.seed, config.graph_dataset.name, config.n_demand_clique,
    #                                                                           config.demand_capacity, config.destruction_quantity)
    # util.save_porting_dictionary(G, path)
    # return

    pg.plot(G, config.graph_path, distribution, config.destruction_precision, dim_ratio,
            config.destruction_show_plot, config.destruction_save_plot, config.seed, "TRU", co.PlotType.TRU, config.destruction_quantity)

    # print(broken_nodes)
    # print(broken_edges)
    # time.sleep(10)
    # return

    # hypothetical routability
    if not is_feasible(G, is_fake_fixed=True):
        print("This instance is not solvable. Check the number of demand edges, theirs and supply links capacity.\n\n\n")
        return None

    # repair demand edges
    demand_node = get_demand_nodes(G)
    for dn in demand_node:
        repair_node(G, dn)
        # INITIAL NODES repairs are not counted in the stats

    iter = 0
    # true ruotability

    routed_flow = 0
    packet_monitor = 0
    monitors_stats = set()

    if config.monitoring_type == co.PriorKnowledge.FULL:
        gain_knowledge_all(G)

    assert(config.monitors_budget >= len(get_demand_nodes(G)))

    # set as monitors all the nodes that are demand endpoints
    monitors_map = defaultdict(list)
    for n1, n2, _ in get_demand_edges(G):
        G.nodes[n1][co.ElemAttr.IS_MONITOR.value] = True
        G.nodes[n2][co.ElemAttr.IS_MONITOR.value] = True
        monitors_stats |= {n1, n2}

        monitors_map[n1].append((n1, n2))
        monitors_map[n2].append((n1, n2))

    pr_on_s3, pr_on_s4 = 0, 0
    # START
    # TODO: azzoppare cedar con routing nostro
    # TODO: is routable according to routing algorithm! Not the system of equations
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
                 "packet_monitoring": packet_monitor}

        # -------------- 0. Monitor placement --------------

        if config.protocol_monitor_placement == co.ProtocolMonitorPlacement.BUDGET_W_REPLACEMENT:
            monitors_map = mon.removing_monitor(G, monitors_map, config)
            monitors, monitors_repaired, candidate_monitors_dem = mon.new_monitoring_add(G, config)
            monitors_map = mon.merge_monitor_maps(monitors_map, candidate_monitors_dem)
            # stats["node"] += monitors_repaired

        elif config.protocol_monitor_placement == co.ProtocolMonitorPlacement.STEP_BY_STEP:
            monitors = mon.original_monitoring_add(G, config)
            stats["monitors"] |= monitors
            monitors_stats = stats["monitors"]

        # no monitor for ORACLE mode

        # -------------- 1. Tomography --------------
        if config.monitoring_type == co.PriorKnowledge.TOMOGRAPHY:
            monitoring = gain_knowledge_tomography(G,
                                                   stats["packet_monitoring"],
                                                   config.monitoring_messages_budget,
                                                   elements_val_id,
                                                   elements_id_val,
                                                   config.UNK_prior,
                                                   monitors_map,
                                                   config.protocol_monitor_placement,
                                                   config.is_exhaustive_paths)

            if monitoring is None:
                stats_list.append(stats)
                return stats_list

            stats_packet_monitoring, demand_edges_to_repair, demand_edges_routed_flow = monitoring
            routed_flow += sum(demand_edges_routed_flow)
            stats["flow"] = routed_flow
            stats["packet_monitoring"] += stats_packet_monitoring
            packet_monitor = stats["packet_monitoring"]

        demand_edges = get_demand_edges(G, is_check_unsatisfied=True, is_residual=True)
        print("> Residual demand edges", len(demand_edges), demand_edges)

        # -------------- 2. Repairing --------------
        paths_proposed = frp.find_paths_to_repair(config.repairing_mode, G, demand_edges_to_repair, is_oracle=config.is_oracle_baseline)
        path_to_fix = frpp.find_path_picker(config.picking_mode, G, paths_proposed, config.repairing_mode, is_oracle=config.is_oracle_baseline)
        fixed_nodes, fixed_edges = do_fix_path(G, path_to_fix)

        stats["node"] += fixed_nodes
        stats["edge"] += fixed_edges
        stats_list.append(stats)

    return stats_list

