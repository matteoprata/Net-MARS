import time

import networkx as nx

import src.plotting.graph_plotting as pg
from src.preprocessing.graph_routability import *
from src.preprocessing.graph_preprocessing import *
from src.preprocessing.graph_monitoring import *
from src.preprocessing.graph_utils import *
import src.constants as co
import pandas as pd
import src.utilities.util_widest_path as mx
import src.utilities.util_routing_stpath as mxv


def save_porting_dictionary(G, fname):
    """ Stores the graph characteristics. """
    demand_edges_flow = {str((n1, n2)): c for n1, n2, c in get_demand_edges(G)}
    normal_edges_flow = {str((n1, n2)): G.edges[n1, n2, tip][co.ElemAttr.CAPACITY.value] for n1, n2, tip in G.edges if
                         tip == co.EdgeType.SUPPLY.value}

    normal_edges_stat = {str((n1, n2)): G.edges[n1, n2, tip][co.ElemAttr.STATE_TRUTH.value] for n1, n2, tip in G.edges if
                         tip == co.EdgeType.SUPPLY.value}
    normal_nodes_stat = {str(n): G.nodes[n][co.ElemAttr.STATE_TRUTH.value] for n in G.nodes}

    out = {"demand_edges_flow": demand_edges_flow,
           "normal_edges_flow": normal_edges_flow,
           "normal_edges_stat": normal_edges_stat,
           "normal_nodes_stat": normal_nodes_stat
           }

    import json
    with open(fname, 'w') as f:
        json.dump(out, f)


def run(config):
    stats_list = []

    # read graph and print stats
    G, elements_val_id, elements_id_val = init_graph(co.PATH_TO_GRAPH, config.graph_path, config.supply_capacity, config)
    print_graph_info(G)

    # normalize coordinates and break components
    dim_ratio = scale_coordinates(G)

    distribution, broken_nodes, broken_edges, perc_broken_elements = destroy(G, config.destruction_type, config.destruction_precision, dim_ratio,
                                                       config.destruction_width, config.n_destruction, config.graph_dataset, config.seed, ratio=config.destruction_uniform_quantity)

    # add_demand_endpoints
    if config.is_demand_clique:
        add_demand_clique(G, config.n_demand_clique, config.demand_capacity)
    else:
        add_demand_pairs(G, config.n_demand_pairs, config.demand_capacity)

    # path = "data/porting/graph-s|{}-g|{}-np|{}-dc|{}-uni-pbro|{}.json".format(config.seed, config.graph_dataset.name, config.n_demand_pairs,
    #                                                              config.demand_capacity, config.destruction_uniform_quantity)
    # save_porting_dictionary(G, path)
    # return

    pg.plot(G, config.graph_path, distribution, config.destruction_precision, dim_ratio,
            config.destruction_show_plot, config.destruction_save_plot, config.seed, "TRU", co.PlotType.TRU)

    # hypothetical routability
    if not is_routable(G, None, is_fake_fixed=True):
        print("This instance is not solvable. Check the number of demand edges, theirs and supply links capacity.\n\n\n")
        return None

    # repair demand edges
    demand_node = get_demand_nodes(G)
    for dn in demand_node:
        repair_node(G, dn)

    iter = 0
    # true ruotability

    routed_flow = 0
    packet_monitor = 0
    monitors_stats = set()

    if config.prior_knowledge == co.PriorKnowledge.FULL:
        gain_knowledge_all(G)

    assert(config.monitors_budget >= len(get_demand_nodes(G)))

    # set as monitors all the nodes that are demand endpoints
    for n1, n2, _ in get_demand_edges(G):
        G.nodes[n1][co.ElemAttr.IS_MONITOR.value] = True
        G.nodes[n2][co.ElemAttr.IS_MONITOR.value] = True

    while not is_routable(G, co.Knowledge.TRUTH):
        iter += 1
        print("\nITER", iter)

        # packet_monitor -- monitors paced up to iteration i
        # monitors -- monitors placed up to now (no duplicates)
        stats = {"iter": iter,
                 "node": [],
                 "edge": [],
                 "flow": 0,
                 "monitors": monitors_stats,
                 "packet_monitoring": packet_monitor}

        # 1. Monitoring
        # set_infinite_weights(G)
        if config.prior_knowledge == co.PriorKnowledge.TOMOGRAPHY:
            stats_monitors, stats_packet_monitoring = gain_knowledge_tomography(G,
                                                                                stats["packet_monitoring"],
                                                                                config.monitoring_messages_budget,
                                                                                elements_val_id,
                                                                                elements_id_val)
            stats["packet_monitoring"] += stats_packet_monitoring
            stats["monitors"] |= stats_monitors  # union !
            packet_monitor = stats["packet_monitoring"]
            monitors_stats = stats["monitors"]

        pg.plot(G, config.graph_path, None, config.destruction_precision, dim_ratio,
                config.destruction_show_plot, config.destruction_save_plot, config.seed, "iter-af{}".format(iter),
                co.PlotType.KNO)

        demand_edges = get_demand_edges(G, is_check_unsatisfied=True)
        print("> Residual demand edges", demand_edges)

        # 1.1 do preliminary pruning of working paths

        # the list of path between demand nodes
        # 2. Compute all shortest paths between demand pairs
        paths = []
        for n1, n2, _ in demand_edges:
            SG = get_supply_graph(G)
            # probabilistic_edge_weights(SG, G)
            path = mxv.widest_path_viv(SG, n1, n2)
            #path = nx.shortest_path(SG, n1, n2, weight=co.ElemAttr.WEIGHT.value, method='dijkstra')  # co.ElemAttr.WEIGHT_UNIT.value
            paths.append(path)

        print(paths)

        # 3. Map the path to its bottleneck capacity
        paths_caps = []
        for path_nodes in paths:
            # min_cap = get_path_cost(G, path_nodes)
            min_cap = get_path_cost_VN(G, path_nodes)  # MINIMIZE expected cost of repair
            paths_caps.append(min_cap)

        # 4. Get the path that maximizes the minimum bottleneck capacity
        path_id_to_fix = np.argmin(paths_caps)
        # max_demand = paths_caps[path_id_to_fix]  # it could also be only the capacity of the demand edge
        print("Selected path has capacity", get_path_residual_capacity(G, paths[path_id_to_fix]))

        # 5. Repair edges and nodes
        path_to_fix = paths[path_id_to_fix]  # 1, 2, 3
        print("> Repairing path", path_to_fix)

        for n1 in path_to_fix:
            did_repair = repair_node(G, n1)
            if did_repair:
                stats["node"].append(n1)

        for i in range(len(path_to_fix) - 1):
            n1, n2 = path_to_fix[i], path_to_fix[i + 1]
            n1, n2 = make_existing_edge(G, n1, n2)
            did_repair = repair_edge(G, n1, n2)
            if did_repair:
                stats["edge"].append((n1, n2))

        # 6. Routing
        # pruning of the max capacity prunable of the demand edge
        d1, d2 = make_existing_edge(G, path_to_fix[0], path_to_fix[-1])
        # st_path = nx.shortest_path(get_supply_graph(G), d1, d2, weight=co.ElemAttr.WEIGHT_UNIT.value, method='dijkstra')
        st_path_cap = get_path_residual_capacity(G, path_to_fix)

        demand_residual = G.edges[d1, d2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value]
        quantity_pruning = min(st_path_cap, demand_residual)

        demand_pruning(G, path_to_fix, quantity_pruning)

        routed_flow += quantity_pruning
        stats["flow"] = routed_flow  # quantity_pruning / len(DEMAND_EDGES) * config.demand_capacity

        # pg.plot(G, config.graph_path, None, config.destruction_precision, dim_ratio,
        #         config.destruction_show_plot, config.destruction_save_plot, config.seed, "iter-af{}".format(iter),
        #         co.PlotType.ROU)

        stats_list.append(stats)

        # 7. Add 1 new monitor, after discovery
        res_demand_edges = get_demand_edges(G, is_check_unsatisfied=True)
        monitor_nodes = get_monitor_nodes(G)
        if len(res_demand_edges) > 0 and len(monitor_nodes) < config.monitors_budget:
            # monitors = monitor_placement_centrality(G, res_demand_edges)
            monitors = monitor_placement_ours(G, res_demand_edges)
            stats["monitors"] |= monitors
            monitors_stats = stats["monitors"]

        print(stats)

    # pg.plot(G, config.graph_path, None, config.destruction_precision, dim_ratio,
    #         config.destruction_show_plot, config.destruction_save_plot, config.seed, "iter{}".format('final'), co.PlotType.KNO)

    return stats_list




