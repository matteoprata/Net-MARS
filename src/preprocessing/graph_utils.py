
import src.constants as co
import numpy as np
import sys
import networkx as nx


# utilities requiring computation
def get_demand_edges(G, is_check_unsatisfied=False, is_capacity=True):
    """node, node, demand"""

    demand_edges = []
    for n1, n2, etype in G.edges:
        if etype == co.EdgeType.DEMAND.value:
            res = G.edges[n1, n2, etype][co.ElemAttr.RESIDUAL_CAPACITY.value]
            if (is_check_unsatisfied and res > 0) or (not is_check_unsatisfied):
                out = (n1, n2, G.edges[n1, n2, etype][co.ElemAttr.CAPACITY.value]) if is_capacity else (n1, n2)
                demand_edges.append(out)
    return demand_edges


def get_supply_edges(G):
    """ Returns only supply edges. """
    return [e for e in G.edges if e[2] == co.EdgeType.SUPPLY.value]


def get_demand_nodes(G):
    demand_nodes = []
    for n1, n2, _ in get_demand_edges(G):
        demand_nodes += [n1, n2]
    return demand_nodes


def is_demand_edge(G, n1, n2):
    return (n1, n2) in get_demand_edges(G, is_capacity=False)


def get_node_degree_working_edges(G, node, is_fake_fixed):
    """ Degree of a node excluding the demand edges."""
    count = 0
    for neig in G.neighbors(node):
        n1, n2 = make_existing_edge(G, neig, node)
        if not is_demand_edge(G, n1, n2):
            is_working_n1 = G.nodes[n1][co.ElemAttr.STATE_TRUTH.value] == co.NodeState.WORKING.value
            is_working_n2 = G.nodes[n2][co.ElemAttr.STATE_TRUTH.value] == co.NodeState.WORKING.value
            is_working_edge = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value] == co.NodeState.WORKING.value
            if (is_working_n1 and is_working_n2 and is_working_edge) or is_fake_fixed:
                count += 1
    return count


def get_supply_graph(G):
    """ The supply graph is G without demand edges. """
    demand_edges = get_demand_edges(G)
    SG = G.copy()
    for n1, n2, _ in demand_edges:
        SG.remove_edge(n1, n2)
    return SG


def get_supply_graph_working_T(G):
    """ The supply graph is G without demand edges. """
    demand_edges = get_demand_edges(G)
    SG = get_supply_graph(G)
    SG_out = SG.copy()
    for n1, n2, et in SG.edges:
        edges_b = get_element_by_state_KT(SG, co.GraphElement.EDGE, co.NodeState.BROKEN, co.Knowledge.TRUTH)
        if (n1, n2, et) in edges_b:
            SG_out.remove_edge(n1, n2, et)

    for n1 in SG.nodes:
        nodes_b = get_element_by_state_KT(SG, co.GraphElement.NODE, co.NodeState.BROKEN, co.Knowledge.TRUTH)
        if n1 in nodes_b:
            SG_out.remove_node(n1)

    return SG_out


def get_capacity_grid(G, knowledge, is_fake_fixed=False):
    """ A dict M[i,j]: residual capacity of edge i,j. """
    grid = dict()
    for n1, n2, etype in G.edges:
        if etype == co.EdgeType.SUPPLY.value:
            if is_fake_fixed:
                grid[n1, n2] = G.edges[n1, n2, etype][co.ElemAttr.CAPACITY.value]
            elif (n1, n2, etype) in get_element_by_state_KT(G, co.GraphElement.EDGE, co.NodeState.WORKING, knowledge):
                grid[n1, n2] = G.edges[n1, n2, etype][co.ElemAttr.RESIDUAL_CAPACITY.value]
            else:
                grid[n1, n2] = 0
    return grid


def get_incident_edges_of_node(node, edges):
    """ returns the list of incident edges to the node, edges are oriented """
    incident_edges = [(n1, n2, cap) for n1, n2, cap in edges if n1 == node or n2 == node]

    to_node = []
    from_node = []

    for i in range(len(incident_edges)):
        id_source, id_target = incident_edges[i][0], incident_edges[i][1]

        edge = (id_source, id_target)
        reverse_edge = (id_target, id_source)

        if edge[0] == node:
            from_node.append(edge)
            to_node.append(reverse_edge)
        else:
            from_node.append(reverse_edge)
            to_node.append(edge)

    return to_node, from_node


def make_existing_edge(G, n1, n2):
    """ Returns the oriented graph, for undirected graphs """
    if (n1, n2) in [(e1, e2) for e1, e2, _ in G.edges]:
        return n1, n2
    else:
        return n2, n1


def repair_node(G, n):
    """ counts the repairs! """
    did_repair = G.nodes[n][co.ElemAttr.STATE_TRUTH.value] == co.NodeState.BROKEN.value
    G.nodes[n][co.ElemAttr.STATE_TRUTH.value] = co.NodeState.WORKING.value
    G.nodes[n][co.ElemAttr.POSTERIOR_BROKEN.value] = co.NodeState.WORKING.value
    return did_repair


def repair_edge(G, n1, n2):
    """ counts the repairs! """
    did_repair = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value] == co.NodeState.BROKEN.value
    G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value] = co.NodeState.WORKING.value
    G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value] = co.NodeState.WORKING.value
    return did_repair


def discover_edge(G, n1, n2, is_working):
    G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value] = 1-is_working
    G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value] = 1-is_working


def discover_node(G, n, is_working):
    G.nodes[n][co.ElemAttr.STATE_TRUTH.value] = 1-is_working
    G.nodes[n][co.ElemAttr.POSTERIOR_BROKEN.value] = 1-is_working


def get_element_by_state_KT(G, graph_element, state, knowledge):
    """ K: discovered, T: truth """
    elements = []

    # conditions to add elements to return
    sat_state_K = lambda state, prob: (state == co.NodeState.BROKEN and prob == 1) or \
                                      (state == co.NodeState.WORKING and prob == 0) or \
                                      (state == co.NodeState.UNK and 0 < prob < 1)

    if graph_element == co.GraphElement.EDGE:
        for n1, n2, _ in get_supply_edges(G):
            prob = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value]
            state_T = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value]

            # returns the elements that are truly of that type
            if knowledge == co.Knowledge.TRUTH:
                if state.value == state_T:
                    elements.append((n1, n2, co.EdgeType.SUPPLY.value))
            elif knowledge == co.Knowledge.KNOW:
                if sat_state_K(state, prob):
                    elements.append((n1, n2, co.EdgeType.SUPPLY.value))

    elif graph_element == co.GraphElement.NODE:
        for n1 in G.nodes:
            prob = G.nodes[n1][co.ElemAttr.POSTERIOR_BROKEN.value]
            state_T = G.nodes[n1][co.ElemAttr.STATE_TRUTH.value]

            if knowledge == co.Knowledge.TRUTH:
                if state.value == state_T:
                    elements.append(n1)
            elif knowledge == co.Knowledge.KNOW:
                if sat_state_K(state, prob):
                    elements.append(n1)
    return elements


def get_path_residual_capacity(G, path_nodes):
    """ The capacity of the path is the minimum residual capacity of the edges of the path. """

    capacities = []
    for i in range(len(path_nodes) - 1):
        n1, n2 = path_nodes[i], path_nodes[i + 1]
        n1, n2 = make_existing_edge(G, n1, n2)
        cap = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value]  # TODO
        capacities.append(cap)
    min_cap = min(capacities)
    return min_cap


def demand_pruning(G, path, quantity):
    """ Prune the edges capacity by quantity, assumed to be small enough. """
    d1, d2 = make_existing_edge(G, path[0], path[-1])
    demand_full_capacity = G.edges[d1, d2, co.EdgeType.DEMAND.value][co.ElemAttr.CAPACITY.value]
    G.edges[d1, d2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value] -= quantity

    for i in range(len(path) - 1):
        n1, n2 = path[i], path[i + 1]
        n1, n2 = make_existing_edge(G, n1, n2)
        full_cap = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.CAPACITY.value]
        cap = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value]
        assert(cap-quantity >= 0)
        G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value] -= quantity

        # debugging
        G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.SAT_DEM.value][(d1, d2)] += round(quantity/full_cap, 3)  # (demand edge): percentage flow
        G.edges[d1, d2, co.EdgeType.DEMAND.value][co.ElemAttr.SAT_SUP][(n1, n2)] += round(quantity/demand_full_capacity, 3)  # (demand edge): percentage flow


def demand_node_position(demand_edges, demands, nodes):
    """ Maps every node to their position in the graph wrt the demand graph 0 source, 1 mid, 2 target. """
    demand_node_pos = {}

    for node in nodes:
        for h in demands:
            demand_node_pos[node, h] = 1  # 1 nodo centrale di demand

    sources = [i for i, _, _ in demand_edges]
    targets = [j for _, j, _ in demand_edges]

    for i, h in enumerate(demands):
        demand_node_pos[sources[i], h] = 0  # 0 nodo sorgente di demand
        demand_node_pos[targets[i], h] = 2  # 2 nodo target di demand

    return demand_node_pos


def infinite_edge_weights(G):
    for e1, e2, et in G.edges:
        if G.edges[e1, e2, et][co.ElemAttr.RESIDUAL_CAPACITY.value] == 0:
            G.edges[e1, e2, et][co.ElemAttr.WEIGHT.value] = np.inf
        else:
            G.edges[e1, e2, et][co.ElemAttr.WEIGHT.value] = 1


def element_state_wprob(element_prob):
    if element_prob == 0:
        return co.NodeState.WORKING
    elif element_prob == 1:
        return co.NodeState.BROKEN
    elif 0 < element_prob < 1:
        return co.NodeState.UNK


def get_element_state(G, n1, n2, knowledge):
    if n2 is None:  # element is an edge
        if knowledge == co.Knowledge.TRUTH:
            return G.nodes[n1][co.ElemAttr.STATE_TRUTH.value]
        else:
            return  element_state_wprob(G.nodes[n1][co.ElemAttr.POSTERIOR_BROKEN.value])
    else:
        if knowledge == co.Knowledge.TRUTH:
            return G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value]
        else:
            return element_state_wprob(G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value])


def get_path_cost_VN(G, path_nodes):
    """ VERSIONE NUOVA: returns the expected repair cost. """
    cap = get_path_residual_capacity(G, path_nodes)

    cost_broken_els = 0
    cost_broken_els_exp = 0

    # expected cost of repairing the nodes
    for n1 in path_nodes:
        posterior_broken_node = G.nodes[n1][co.ElemAttr.POSTERIOR_BROKEN.value]
        if posterior_broken_node == 1:  # broken
            cost_broken_els += co.repair_cost
        elif 0 < posterior_broken_node < 1:  # unknown
            cost_broken_els_exp += co.repair_cost * posterior_broken_node

    # expected cost of repairing the edges
    for i in range(len(path_nodes) - 1):
        n1, n2 = path_nodes[i], path_nodes[i + 1]
        n1, n2 = make_existing_edge(G, n1, n2)
        posterior_broken_edge = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value]
        if posterior_broken_edge == 1:  # broken
            cost_broken_els += co.repair_cost
        elif 0 < posterior_broken_edge < 1:  # unknown
            cost_broken_els_exp += co.repair_cost * posterior_broken_edge

    return (cost_broken_els + cost_broken_els_exp) / (cap + co.epsilon)


def get_path_cost_VO(G, path_nodes):
    """ VERSIONE VECCHIA, old cedar, va peggio: Returns the expected repair cost old version. """
    cap = get_path_residual_capacity(G, path_nodes)

    cost_broken_els = 0
    cost_broken_els_exp = 0

    # expected cost of repairing the nodes
    for n1 in path_nodes:
        posterior_broken_node = G.nodes[n1][co.ElemAttr.POSTERIOR_BROKEN.value]
        if 0 < posterior_broken_node <= 1:  # broken - unknown
            cost_broken_els_exp += co.repair_cost

    # expected cost of repairing the edges
    for i in range(len(path_nodes) - 1):
        n1, n2 = path_nodes[i], path_nodes[i + 1]
        n1, n2 = make_existing_edge(G, n1, n2)
        posterior_broken_edge = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value]
        if 0 < posterior_broken_edge <= 1:  # broken - unknown
            cost_broken_els_exp += co.repair_cost

    return cap / (cost_broken_els + cost_broken_els_exp + co.repair_cost)


def probabilistic_edge_weights(SG, G):
    for n1, n2, et in SG.edges:
        n1, n2 = make_existing_edge(G, n1, n2)

        edge_cap = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value]
        edge_known = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value]
        broken_edge = 0
        if edge_known == co.NodeState.BROKEN.value:
            broken_edge += co.repair_cost
        elif 1 > edge_known > 0:
            broken_edge += co.repair_cost * edge_known

        n1_known = G.nodes[n1][co.ElemAttr.POSTERIOR_BROKEN.value]
        broken_n1 = 0
        if n1_known == co.NodeState.BROKEN.value:
            broken_n1 += co.repair_cost
        elif 1 > n1_known > 0:
            broken_n1 += co.repair_cost * n1_known

        n2_known = G.nodes[n2][co.ElemAttr.POSTERIOR_BROKEN.value]
        broken_n2 = 0
        if n2_known == co.NodeState.BROKEN.value:
            broken_n2 += co.repair_cost
        elif 1 > n2_known > 0:
            broken_n2 += co.repair_cost * n2_known

        # weight = np.inf if edge_cap == 0 else (broken_edge + (broken_n1 + broken_n2) / 2) * (1/(edge_cap + co.epsilon) + 1)  # 1 stands for the crossing
        weight = np.inf if edge_cap == 0 else (broken_edge + (broken_n1 + broken_n2) / 2) #* (1/(edge_cap + co.epsilon))  # 1 stands for the crossing

        SG.edges[n1, n2, et][co.ElemAttr.WEIGHT.value] = weight
        G.edges[n1, n2, et][co.ElemAttr.WEIGHT.value] = weight


def broken_elements_in_path(G, path_nodes):
    """ Returns the broken or unknown elements in path. """
    n_broken_unk = 0
    for i in range(len(path_nodes) - 1):
        n1, n2 = path_nodes[i], path_nodes[i + 1]
        n1, n2 = make_existing_edge(G, n1, n2)
        if G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value] > 0:
            n_broken_unk += 1
    return n_broken_unk


def set_infinite_weights(G):
    """ Sets to infinity all the weights of the edges associated to path. """
    for n1, n2, et in get_supply_edges(G):
        if G.edges[n1, n2, et][co.ElemAttr.RESIDUAL_CAPACITY.value] == 0:
            G.edges[n1, n2, et][co.ElemAttr.WEIGHT.value] = np.inf
            G.edges[n1, n2, et][co.ElemAttr.WEIGHT_UNIT.value] = np.inf
