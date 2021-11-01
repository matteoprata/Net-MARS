
from src.preprocessing.graph_utils import *
import networkx as nx
import sys
np.set_printoptions(threshold=sys.maxsize)

import tqdm

# OK
# lists of edges are not symmetric, check symmetric edge every time


def broken_paths_without_n(G, paths, broken_paths, broken_edges_T, broken_nodes_T, elements_val_id, elements_id_val, element=None):
    """ Returns the set of paths by ID that contain at least 1 broken element or that pass through a node element. """

    if broken_paths is None:
        broken_paths = []
        for path_nodes in paths:
            # adds a path to broken_paths if a node or an edge of the path is broken, i.e. the path is broken
            is_path_broken = False

            # check if path broken
            for i in range(len(path_nodes) - 1):
                n1, n2 = path_nodes[i], path_nodes[i + 1]
                is_path_broken = n1 in broken_nodes_T or n2 in broken_nodes_T or (n1, n2) in broken_edges_T or (n2, n1) in broken_edges_T
                if is_path_broken:
                    break

            # if the path was working, then set all as discovered working
            if not is_path_broken:
                for i in range(len(path_nodes) - 1):
                    n1, n2 = path_nodes[i], path_nodes[i + 1]
                    n1, n2 = make_existing_edge(G, n1, n2)
                    discover_node(G, n1, 1)
                    discover_node(G, n2, 1)
                    discover_edge(G, n1, n2, 1)
            else:
                # if the path is not working and there is only one unknown or broken element, set it as discovered broken
                broken_unk_path_node = [n for n in path_nodes if G.nodes[n][co.ElemAttr.POSTERIOR_BROKEN.value] > 0]  # uncertainty or broken

                broken_unk_path_edge = []
                for i in range(len(path_nodes) - 1):
                    n1, n2 = path_nodes[i], path_nodes[i + 1]
                    n1, n2 = make_existing_edge(G, n1, n2)

                    try:
                        if G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value] > 0:
                            broken_unk_path_edge.append((n1, n2))
                    except:
                        print("ERROR")
                        print(get_demand_edges(G))
                        print(n1, n2)
                        exit()

                if len(broken_unk_path_node) + len(broken_unk_path_edge) == 1:  # single element is broken or unknown
                    if len(broken_unk_path_node) == 1:
                        bun = broken_unk_path_node[0]
                        discover_node(G, bun, 0)
                    else:
                        bue = broken_unk_path_node[0]
                        n1, n2 = make_existing_edge(G, bue[0], bue[1])
                        discover_edge(G, n1, n2, 0)

                path_els = [elements_val_id[n] for n in path_nodes]
                path_els += [elements_val_id[make_existing_edge(G, path_nodes[i], path_nodes[i + 1])] for i in range(len(path_nodes) - 1)]
                broken_paths.append(path_els)

    paths_out = broken_paths[:]
    remove_ids = []
    if element is not None:
        # saves the index of the path to remove
        for i in range(len(broken_paths)):
            if elements_val_id[element] in broken_paths[i]:
                remove_ids.append(i)

        paths_out = []
        # keeps the paths that are not traversed by 'element'
        for i in range(len(broken_paths)):
            if i not in remove_ids:
                paths_out.append(broken_paths[i])

    return paths_out


def gain_knowledge_of_n(SG, element, element_type, broken_paths, broken_paths_padded, original_posterior, tot_els, paths, broken_edges_T,
                        broken_nodes_T, elements_val_id, elements_id_val):
    bpwn = broken_paths_without_n(SG, paths, broken_paths, broken_edges_T, broken_nodes_T, elements_val_id, elements_id_val, element=element)

    if len(bpwn) == 0:
        return original_posterior

    broken_paths_without_n_padded = np.zeros(shape=(len(bpwn), tot_els))
    for i, p in enumerate(bpwn):
        broken_paths_without_n_padded[i, :len(p)] = p

    working_edges_P = get_element_by_state_KT(SG, co.GraphElement.EDGE, co.NodeState.WORKING, co.Knowledge.KNOW)
    working_nodes_P = get_element_by_state_KT(SG, co.GraphElement.NODE, co.NodeState.WORKING, co.Knowledge.KNOW)

    working_elements_ids = [elements_val_id[e[:2]] for e in working_edges_P] + [elements_val_id[n] for n in working_nodes_P]

    a_prior = SG.edges[element[0], element[1], co.EdgeType.SUPPLY.value][co.ElemAttr.PRIOR_BROKEN.value] if element_type == co.GraphElement.EDGE else SG.nodes[element][co.ElemAttr.PRIOR_BROKEN.value]
    num = probability_broken(broken_paths_without_n_padded, a_prior, working_elements_ids)
    den = probability_broken(broken_paths_padded, a_prior, working_elements_ids)
    prob_n_broken = a_prior * num / den
    assert(prob_n_broken >= 0)
    return prob_n_broken


def gain_knowledge_of_all(G, elements_val_id, elements_id_val):
    demand_edges = get_demand_edges(G, is_check_unsatisfied=True)
    broken_edges_T = get_element_by_state_KT(G, co.GraphElement.EDGE, co.NodeState.BROKEN, co.Knowledge.TRUTH)
    broken_nodes_T = get_element_by_state_KT(G, co.GraphElement.NODE, co.NodeState.BROKEN, co.Knowledge.TRUTH)

    # the list of path between demand nodes
    paths = []
    for n1, n2, _ in demand_edges:
        SG = get_supply_graph(G)
        probabilistic_edge_weights(SG)
        path = nx.shortest_path(SG, n1, n2, weight=co.ElemAttr.WEIGHT.value, method='dijkstra')
        paths.append(path)

    n_edges, n_nodes = len(G.edges), len(G.nodes)
    tot_els = n_edges + n_nodes

    # broken_paths
    bp = broken_paths_without_n(G, paths, None, broken_edges_T, broken_nodes_T, elements_val_id, elements_id_val, element=None)
    bp_padded = np.zeros(shape=(len(bp), tot_els))
    for i, p in enumerate(bp):
        bp_padded[i, :len(p)] = p

    # Assign a probability to every element
    pars = tot_els, paths, broken_edges_T, broken_nodes_T, elements_val_id, elements_id_val
    for n1 in tqdm.tqdm(G.nodes):
        original_posterior = G.nodes[n1][co.ElemAttr.POSTERIOR_BROKEN.value]
        if original_posterior not in [0, 1]:
            G.nodes[n1][co.ElemAttr.POSTERIOR_BROKEN.value] = gain_knowledge_of_n(G, n1, co.GraphElement.NODE, bp, bp_padded, original_posterior, *pars)

    for n1, n2, gt in tqdm.tqdm(G.edges):
        if gt == co.EdgeType.SUPPLY.value:
            original_posterior = G.edges[n1, n2, gt][co.ElemAttr.POSTERIOR_BROKEN.value]
            if original_posterior not in [0, 1]:
                G.edges[n1, n2, gt][co.ElemAttr.POSTERIOR_BROKEN.value] = gain_knowledge_of_n(G, (n1, n2), co.GraphElement.EDGE, bp, bp_padded, original_posterior, *pars)


def probability_broken(paths, prior, working_els):
    # TODO: remove rows with same elements
    """ Viviana version! """

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


