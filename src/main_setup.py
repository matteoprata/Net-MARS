
import networkx as nx

import src.plotting.graph_plotting as pg
from src.preprocessing.graph_routability import *
from src.preprocessing.graph_preprocessing import *
from src.preprocessing.graph_monitoring import *
from src.preprocessing.graph_utils import *
import src.constants as co


def run(config):

    # read graph and print stats
    G, elements_val_id, elements_id_val = init_graph(co.path_to_graph, config.graph_name, config.supply_capacity)
    print_graph_info(G)

    # normalize coordinates and break components
    dim_ratio = scale_coordinates(G)

    distribution, broken_nodes, broken_edges = destroy(G, config.destruction_type, config.destruction_precision, dim_ratio,
                                                       config.destruction_width, config.n_destruction, ratio=config.destruction_uniform_quantity)

    # add_demand_endpoints
    if config.is_demand_clique:
        add_demand_clique(G, config.n_demand_clique, config.demand_capacity)
    else:
        add_demand_pairs(G, config.n_demand_pairs, config.demand_capacity)

    # plot graph
    pg.plot(G, config.graph_name, distribution, config.destruction_precision, dim_ratio,
               config.destruction_show_plot, config.destruction_save_plot, config.seed, "TRU", co.PlotType.TRU)

    # hypothetical routability
    if not is_routable(G, None, is_fake_fixed=True):
        print("This instance is not solvable.")
        exit()

    # repair demand edges
    demand_node = get_demand_nodes(G)
    for dn in demand_node:
        repair_node(G, dn)

    iter = 0
    # true ruotability
    while not is_routable(G, co.Knowledge.TRUTH):
        iter += 1
        print("ITER", iter)

        # if len(get_demand_edges(G, is_check_unsatisfied=True)) > 1:
        gain_knowledge_of_all(G, elements_val_id, elements_id_val)  # monitoring

        pg.plot(G, config.graph_name, None, config.destruction_precision, dim_ratio,
                config.destruction_show_plot, config.destruction_save_plot, config.seed, "iter-af{}".format(iter),
                co.PlotType.KNO)

        demand_edges = get_demand_edges(G, is_check_unsatisfied=True)
        print("> Residual demand edges", demand_edges)

        # the list of path between demand nodes
        paths = []
        for n1, n2, _ in demand_edges:
            SG = get_supply_graph(G)
            probabilistic_edge_weights(SG, G)
            path = nx.shortest_path(SG, n1, n2, weight=co.ElemAttr.WEIGHT.value, method='dijkstra')
            paths.append(path)

        # TODO ADD INFEASIBILITY of the paths

        # map the path to its cost
        paths_caps = []
        for path_nodes in paths:
            min_cap = get_path_capacity(G, path_nodes)
            paths_caps.append(min_cap)

        # get the path that maximizes the minimum capacity
        path_id_to_fix = np.argmax(paths_caps)
        max_demand = paths_caps[path_id_to_fix]  # it could also be only the capacity of the demand edge

        # repair edges and nodes
        path_to_fix = paths[path_id_to_fix]  # 1, 2, 3
        for i in range(len(path_to_fix) - 1):
            n1, n2 = path_to_fix[i], path_to_fix[i + 1]
            n1, n2 = make_existing_edge(G, n1, n2)
            repair_edge(G, n1, n2)

        for n1 in path_to_fix:
            repair_node(G, n1)

        print("> Repairing path", path_to_fix)
        d1, d2 = make_existing_edge(G, path_to_fix[0], path_to_fix[-1])
        demand_residual = G.edges[d1, d2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value]
        quantity_pruning = min(max_demand, demand_residual)
        demand_pruning(G, path_to_fix, quantity_pruning)

        pg.plot(G, config.graph_name, None, config.destruction_precision, dim_ratio,
                config.destruction_show_plot, config.destruction_save_plot, config.seed, "iter-af{}".format(iter),
                co.PlotType.ROU)

    pg.plot(G, config.graph_name, None, config.destruction_precision, dim_ratio,
            config.destruction_show_plot, config.destruction_save_plot, config.seed, "iter{}".format('final'), co.PlotType.KNO)
