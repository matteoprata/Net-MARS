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


def run(config):
    stats_list = []

    # read graph and print stats
    G, elements_val_id, elements_id_val = init_graph(co.path_to_graph, config.graph_path, config.supply_capacity, config)
    print_graph_info(G)

    # normalize coordinates and break components
    dim_ratio = scale_coordinates(G)

    distribution, broken_nodes, broken_edges = destroy(G, config.destruction_type, config.destruction_precision, dim_ratio,
                                                       config.destruction_width, config.n_destruction, config.graph_dataset, config.seed, ratio=config.destruction_uniform_quantity)

    # add_demand_endpoints
    if config.is_demand_clique:
        DEMAND_NODES, DEMAND_EDGES = add_demand_clique(G, config.n_demand_clique, config.demand_capacity)
    else:
        DEMAND_NODES, DEMAND_EDGES = add_demand_pairs(G, config.n_demand_pairs, config.demand_capacity)

    # write demand for porting TODO: remove
    # pd.DataFrame(DEMAND_EDGES).to_csv("data/porting/demand-s|{}-g|{}-np|{}-dc|{}.csv".format(config.seed,
    #                                                                                          config.graph_dataset.name,
    #                                                                                          config.n_demand_pairs,
    #                                                                                          config.demand_capacity))

    # plot graph
    pg.plot(G, config.graph_path, distribution, config.destruction_precision, dim_ratio,
            config.destruction_show_plot, config.destruction_save_plot, config.seed, "TRU", co.PlotType.TRU)

    # hypothetical routability
    if not is_routable(G, None, is_fake_fixed=True):
        print("This instance is not solvable. Check the number of demand edges, theirs and supply links capacity.")
        exit()

    # repair demand edges
    demand_node = get_demand_nodes(G)
    for dn in demand_node:
        repair_node(G, dn)

    iter = 0
    # true ruotability
    routed_flow = 0

    if config.prior_knowledge == co.PriorKnowledge.FULL:
        gain_knowledge_all(G)

    while not is_routable(G, co.Knowledge.TRUTH):
        iter += 1
        print("\nITER", iter)
        stats = {"iter": iter, "node": [], "edge": [], "flow": routed_flow}

        # 1. Monitoring
        # set_infinite_weights(G)
        if config.prior_knowledge == co.PriorKnowledge.TOMOGRAPHY:
            gain_knowledge_tomographically(G, elements_val_id, elements_id_val)

        pg.plot(G, config.graph_path, None, config.destruction_precision, dim_ratio,
                config.destruction_show_plot, config.destruction_save_plot, config.seed, "iter-af{}".format(iter),
                co.PlotType.KNO)

        demand_edges = get_demand_edges(G, is_check_unsatisfied=True)
        print("> Residual demand edges", demand_edges)

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
            min_cap = get_path_cost_VN(G, path_nodes)
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

        # ROUTING
        # pruning of the max capacity prunable of the demand edge
        d1, d2 = make_existing_edge(G, path_to_fix[0], path_to_fix[-1])
        # st_path = nx.shortest_path(get_supply_graph(G), d1, d2, weight=co.ElemAttr.WEIGHT_UNIT.value, method='dijkstra')
        st_path_cap = get_path_residual_capacity(G, path_to_fix)

        demand_residual = G.edges[d1, d2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value]
        quantity_pruning = min(st_path_cap, demand_residual)

        demand_pruning(G, path_to_fix, quantity_pruning)

        routed_flow += quantity_pruning
        stats["flow"] = routed_flow  # quantity_pruning / len(DEMAND_EDGES) * config.demand_capacity

        pg.plot(G, config.graph_path, None, config.destruction_precision, dim_ratio,
                config.destruction_show_plot, config.destruction_save_plot, config.seed, "iter-af{}".format(iter),
                co.PlotType.ROU)

        stats_list.append(stats)

    pg.plot(G, config.graph_path, None, config.destruction_precision, dim_ratio,
            config.destruction_show_plot, config.destruction_save_plot, config.seed, "iter{}".format('final'), co.PlotType.KNO)

    return stats_list
