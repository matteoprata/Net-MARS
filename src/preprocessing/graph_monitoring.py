
from src.preprocessing.graph_utils import *
import networkx as nx
import sys
np.set_printoptions(threshold=sys.maxsize)


def foo(paths, broken_edges, broken_nodes, elements_val_id, elements_id_val, element=None):
    """ Returns the set of paths that contain at least 1 broken element or that pass through a node element. """

    paths_broken_sids = []
    for path_nodes in paths:
        # adds a path to broken_paths if a node or an edge of the path is broken
        for i in range(len(path_nodes) - 1):
            n1, n2 = path_nodes[i], path_nodes[i + 1]
            is_broken_path = n1 in broken_nodes or n2 in broken_nodes or (n1, n2) in broken_edges or (n2, n1) in broken_edges

            if is_broken_path:
                broken_path_els = [elements_val_id[n] for n in path_nodes]
                broken_path_els += [elements_val_id[(path_nodes[i], path_nodes[i + 1])] for i in range(len(path_nodes) - 1)]
                paths_broken_sids.append(broken_path_els)
                break

    remove_ids = []
    if element is not None:
        for i in range(len(paths_broken_sids)):
            if elements_id_val[element] not in paths_broken_sids[i]:
                remove_ids.append(i)

    paths_out = []
    for i in range(len(paths_broken_sids)):
        if i not in remove_ids:
            paths_out.append(paths_broken_sids[i])

    return paths_out


def monitor_gain_knowledge(G, elements_val_id, elements_id_val):
    demand_edges = get_demand_edges(G)
    SG = get_supply_graph(G, demand_edges)
    broken_edges, broken_nodes, working_edges, working_nodes = get_broken_elements(SG)
    del G

    print(broken_edges)
    print(broken_nodes)

    paths = []
    # the list of path between demand nodes
    for n1, n2, _ in demand_edges:
        path = nx.shortest_path(SG, n1, n2, weight=None, method='dijkstra')
        paths.append(path)

    n_edges, n_nodes = len(SG.edges), len(SG.nodes)
    tot_els = n_edges + n_nodes

    id = 18
    broken_paths = foo(paths, broken_edges, broken_nodes, elements_val_id, elements_id_val, element=None)
    broken_paths_through_n = foo(paths, broken_edges, broken_nodes, elements_val_id, elements_id_val, element=id)

    broken_paths_padded = np.zeros(shape=(len(broken_paths), tot_els))
    broken_paths_through_n_padded = np.zeros(shape=(len(broken_paths_through_n), tot_els))

    for i, p in enumerate(broken_paths):
        broken_paths_padded[i, :len(p)] = p

    for i, p in enumerate(broken_paths_through_n):
        broken_paths_through_n_padded[i, :len(p)] = p

    working_elements_ids = [elements_val_id[e] for e in working_edges] + [elements_val_id[n] for n in working_nodes]

    num = probability_broken(broken_paths_through_n_padded, 0.5, working_elements_ids)
    den = probability_broken(broken_paths_padded, 0.5, working_elements_ids)
    prob_n_broken = .5 * num / den
    print(prob_n_broken)
    exit()


def probability_broken(paths, prior, working_els):
    if len(working_els) > 0:
        paths[np.isin(paths, working_els)] = 0

    n_paths = paths.shape[0]
    if n_paths == 0:
        return -1

    if n_paths == 1:
        non_zeros_count = np.sum(paths[paths != 0])
        if non_zeros_count == 0:
            return 0
        return 1 - (1 - prior) ** non_zeros_count

    elif n_paths == 2:
        p1, p2 = paths[0, :], paths[1, :]
        p1p2 = np.array(list(set(p1).union(set(p2))))

        l1, l2, l12 = np.sum(p1[p1 > 0]), np.sum(p2[p2 > 0]), np.sum(p1p2[p1p2 > 0])
        return 1 - (1 - prior) ** l1 - (1 - prior) ** l2 + (1 - prior) ** l12

    else:
        target = paths[:-1, :]
        target[np.isin(target, paths[-1, :])] = 0
        non_zeros_count_last = np.sum(paths[-1, :] > 0)

        prob = probability_broken(paths[:-1, :], prior, working_els) - probability_broken(target, prior, working_els) * (1 - prior) ** non_zeros_count_last
        return prob
