
import src.constants as co
import numpy as np
import sys
import networkx as nx
import src.utilities.util_routing_stpath as mxv
import src.preprocessing.graph_routability as grout
import tqdm


# utilities requiring computation
def get_demand_edges(G, is_check_unsatisfied=False, is_capacity=True, is_residual=False):
    """ node, node, demand.
    is_check_unsatisfied: keeps edges only if the residual capacity i > 0;
    is_capacity: show third component capacity
    is_residual: is residual demand or total demand """

    demand_edges = []
    for n1, n2, etype in G.edges:
        if etype == co.EdgeType.DEMAND.value:
            res = G.edges[n1, n2, etype][co.ElemAttr.RESIDUAL_CAPACITY.value]
            if (is_check_unsatisfied and res > 0) or (not is_check_unsatisfied):
                attr = co.ElemAttr.RESIDUAL_CAPACITY.value if is_residual else co.ElemAttr.CAPACITY.value
                out = (n1, n2, G.edges[n1, n2, etype][attr]) if is_capacity else (n1, n2)
                demand_edges.append(out)
    return demand_edges


def get_supply_edges(G):
    """ Returns only supply edges. """
    return [e for e in G.edges if e[2] == co.EdgeType.SUPPLY.value]


def get_supply_graph_diameter(SG):
    return nx.diameter(SG)


def get_supply_max_capacity(config):
    return max(config.supply_capacity[1]-1, config.backbone_capacity)   # TODO careful


def get_demand_nodes(G, is_residual=False):
    demand_nodes = set()
    for n1, n2, _ in get_demand_edges(G):
        if not is_residual or G.edges[n1, n2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value] > 0:
            demand_nodes |= {n1, n2}
    return demand_nodes


# ---- MONITORS

def do_k_monitoring(G, monitor, K=2):
    if K > 0:
        SG = get_supply_graph(G)
        reach_k_paths = nx.single_source_shortest_path(SG, monitor, cutoff=K)
        monitoring_messages = 0
        for no in reach_k_paths:
            discover_path_truth_limit_broken(G, reach_k_paths[no])
            monitoring_messages += 1
        return monitoring_messages
    return 0


def get_monitor_nodes(G):
    """ The set of monitor nodes."""
    monitors = []
    for n in G.nodes:
        if co.ElemAttr.IS_MONITOR.value in G.nodes[n].keys() and \
                G.nodes[n][co.ElemAttr.IS_MONITOR.value]:
            monitors.append(n)
    return monitors






def is_demand_edge(G, n1, n2):
    return (n1, n2) in get_demand_edges(G, is_capacity=False)


# def is_there_worcap_path(G, n1o, n2o):
#     """ Assumes one passes a supply graph only. NOT WORKING nodes or edges, SATURATED edges."""
#     GCO = G.copy()
#     nodes = list(GCO.nodes)[:]
#     for n in nodes:
#         if GCO.nodes[n][co.ElemAttr.STATE_TRUTH.value] == co.NodeState.BROKEN.value:
#             GCO.remove_node(n)
#
#     edges = list(GCO.edges)[:]
#     for n1, n2, _ in edges:
#         if GCO.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value] == co.NodeState.BROKEN.value or \
#                 GCO.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value] == 0:
#             GCO.remove_edge(n1, n2)
#     try:
#         nx.shortest_path(GCO, n1o, n2o)
#         return True
#     except:
#         return False


def is_there_working_path(G, n1o, n2o):
    """ Assumes one passes a supply graph only. NOT WORKING nodes or edges, SATURATED edges."""
    GCO = G.copy()
    nodes = list(GCO.nodes)[:]
    for n in nodes:
        if GCO.nodes[n][co.ElemAttr.STATE_TRUTH.value] == co.NodeState.BROKEN.value:
            GCO.remove_node(n)

    edges = list(GCO.edges)[:]
    for n1, n2, _ in edges:
        if GCO.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value] == co.NodeState.BROKEN.value:
            GCO.remove_edge(n1, n2)
    try:
        nx.shortest_path(GCO, n1o, n2o)
        return True
    except:
        return False


def is_known_path(G, path):
    """ Return True iff the path is fully known. """
    for i in range(len(path) - 1):
        n1, n2 = make_existing_edge(path[i], path[i + 1])
        pun_e = 1 > G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value] > 0
        pun_n1 = 1 > G.nodes[n1][co.ElemAttr.POSTERIOR_BROKEN.value] > 0
        pun_n2 = 1 > G.nodes[n2][co.ElemAttr.POSTERIOR_BROKEN.value] > 0

        if pun_e or pun_n1 or pun_n2:
            return False
    return True


def heuristic_priority_pruning(G, d1, d2):
    """ Priority is the average (over the endpoints) of the ratio demand edge residual capacity over sum of residual
    capacities over the neighbouring nodes. """

    SG = get_supply_graph(G)
    res_d1, res_d2 = 0, 0
    demand_res_cap = G.edges[d1, d2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value]

    for ne in SG.neighbors(d1):
        ns, nd = make_existing_edge(ne, d1)
        res_d1 += G.edges[ns, nd, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value]

    for ne in SG.neighbors(d2):
        ns, nd = make_existing_edge(ne, d2)
        res_d2 += G.edges[ns, nd, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value]

    priority = (demand_res_cap/res_d1 + demand_res_cap/res_d2) / 2    # TODO remove / 2, choose min
                                                                      # || find couple of endpoint which has the min prob.
    return priority


def heuristic_priority_pruning_V2(G, d1, d2, path):
    """ Priority is the average (over the endpoints) of the ratio demand edge residual capacity over sum of residual
    capacities over the neighbouring nodes. """
    cap_path = get_path_residual_capacity(G, path)
    cap_dem = G.edges[d1, d2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value]

    return cap_dem / cap_path


def is_bubble(G, path):
    """ Condition 1 only the endpoint demand edges are the demand edges within the path.
        Condition 2, all nodes in the path (except the endpoints) must have only connections within the set of nodes of the path. """

    SG = get_supply_graph(G)
    d1, d2 = path[0], path[-1]
    d1, d2 = make_existing_edge(d1, d2)
    dem_nodes = get_demand_nodes(SG, is_residual=True)

    tuple_is_unique = len(set(path).intersection(dem_nodes - {d1, d2})) == 0

    for i in range(len(path)-2):
        node = path[i+1]
        for ne in SG.neighbors(node):
            n1, n2 = make_existing_edge(node, ne)
            cap = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value]
            # TODO check if broken, if so, not a bubble
            if ne not in path and cap > 0:
                return False
    return tuple_is_unique


def do_prune(G, path, is_log=True):
    """ Assumes that the path is working! """
    d1, d2 = path[0], path[-1]
    d1, d2 = make_existing_edge(d1, d2)

    st_path_cap = get_path_residual_capacity(G, path)
    demand_residual = G.edges[d1, d2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value]
    quantity_pruning = min(st_path_cap, demand_residual)
    demand_pruning(G, path, quantity_pruning)
    if is_log:
        print("path exists, pruning quantity:", quantity_pruning, "on edge", d1, d2, "of res capacity", demand_residual, "on path of capacity", st_path_cap)
    return quantity_pruning


def is_worcap_path(G, path):
    """ Assumes one passes a supply graph only. NOT WORKING nodes or edges, SATURATED edges."""

    for i in range(len(path) - 1):
        n1, n2 = path[i], path[i + 1]
        n1, n2 = make_existing_edge(n1, n2)
        state_n1 = G.nodes[n1][co.ElemAttr.STATE_TRUTH.value]
        state_n2 = G.nodes[n2][co.ElemAttr.STATE_TRUTH.value]
        state_n1n2 = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value]
        capacity_n1n2 = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value]
        assert(capacity_n1n2 > 0)
        if state_n1 or state_n2 or state_n1n2 or capacity_n1n2 <= 0:
            return False
    return True

def get_node_degree_working_edges(G, node, is_fake_fixed):
    """ Degree of a node excluding the demand edges."""
    count = 0
    for neig in G.neighbors(node):
        n1, n2 = make_existing_edge(neig, node)
        if not is_demand_edge(G, n1, n2):
            is_working_n1 = G.nodes[n1][co.ElemAttr.STATE_TRUTH.value] == co.NodeState.WORKING.value
            is_working_n2 = G.nodes[n2][co.ElemAttr.STATE_TRUTH.value] == co.NodeState.WORKING.value
            is_working_edge = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value] == co.NodeState.WORKING.value
            if (is_working_n1 and is_working_n2 and is_working_edge) or is_fake_fixed:
                count += 1
    return count


def is_path_working(G, path):
    for i in range(len(path)-1):
        n1, n2 = path[i], path[i+1]
        n1, n2 = make_existing_edge(n1, n2)
        if G.nodes[n1][co.ElemAttr.STATE_TRUTH.value] == 1 or G.nodes[n2][co.ElemAttr.STATE_TRUTH.value] == 1 or \
                G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value]:
            return False
    return True


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
                grid[n1, n2] = G.edges[n1, n2, etype][co.ElemAttr.RESIDUAL_CAPACITY.value]
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


def is_demand_edge_exists(G, n1, n2):
    return (n1, n2, co.EdgeType.DEMAND.value) in G.edges


def k_hop_discovery(G, source, khop):
    """ Do k-monitoring """
    packet_monitor = 0
    SG = get_supply_graph(G)
    reach_k_paths = nx.single_source_shortest_path(SG, source, cutoff=khop)  # pi: [path nodes]
    for no in reach_k_paths:
        discover_path_truth_limit_broken(G, reach_k_paths[no])
        packet_monitor += 1
    return packet_monitor


def k_hop_destruction(G, source, khop):
    """ Do k-destruction for dynamic tomo-cedar """
    SG = get_supply_graph(G)
    reach_k_paths = nx.single_source_shortest_path(SG, source, cutoff=khop)  # pi: [path nodes]
    destr_nodes, destr_edges = set(), set()
    for no in reach_k_paths:
        path = reach_k_paths[no]
        for i in range(len(path)-1):  # iterate over path nodes
            n1, n2 = make_existing_edge(path[i], path[i + 1])
            if np.random.rand() > G.nodes[n1][co.ElemAttr.RESISTANCE_TO_DESTRUCTION.value]:
                G.nodes[n1][co.ElemAttr.STATE_TRUTH.value] = co.NodeState.BROKEN.value
                destr_nodes.add(n1)

            if np.random.rand() > G.nodes[n2][co.ElemAttr.RESISTANCE_TO_DESTRUCTION.value]:
                G.nodes[n2][co.ElemAttr.STATE_TRUTH.value] = co.NodeState.BROKEN.value
                destr_nodes.add(n2)

            if np.random.rand() > G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESISTANCE_TO_DESTRUCTION.value]:
                G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value] = co.NodeState.BROKEN.value
                destr_edges.add((n1, n2))
    return destr_nodes, destr_edges


def make_existing_edge(n1, n2):
    """ returns the oriented edge, for undirected graphs, assumes that first endpoint is lower. """
    return (n1, n2) if n1 <= n2 else (n2, n1)


def net_max_degree(G):
    return sorted(G.degree, key=lambda x: x[1], reverse=True)[0][1]


def make_components_known_to_state(G, probability_broken: int):
    """ makes all the components state known to probability_broken"""
    for n in G.nodes:
        G.nodes[n][co.ElemAttr.POSTERIOR_BROKEN.value] = probability_broken

    for n1, n2, t in G.edges:
        if t == co.EdgeType.SUPPLY.value:
            G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value] = probability_broken


def make_components_known(G):
    """ makes all the components state known """
    for n in G.nodes:
        G.nodes[n][co.ElemAttr.POSTERIOR_BROKEN.value] = G.nodes[n][co.ElemAttr.STATE_TRUTH.value]

    for n1, n2, t in G.edges:
        if t == co.EdgeType.SUPPLY.value:
            G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value] = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value]

# - - - - - - - - - - REPAIRING - - - - - - - - - -

def do_fix_path(G, path_to_fix):
    """ Fixes the edges and nodes and returns them """
    fixed_edges, fixed_nodes = [], []

    if path_to_fix is not None:

        # NODES
        for n1 in path_to_fix:
            did_repair = do_repair_node(G, n1)
            if did_repair:
                fixed_nodes.append(n1)

        # EDGES
        for i in range(len(path_to_fix) - 1):
            n1, n2 = path_to_fix[i], path_to_fix[i + 1]
            n1, n2 = make_existing_edge(n1, n2)
            did_repair = do_repair_edge(G, n1, n2)
            if did_repair:
                fixed_edges.append((n1, n2))

    return fixed_nodes, fixed_edges


def do_fix_path_smart(G, path_to_fix):
    """ Fixes the edges and nodes and returns them only until there is no path, otherwise it stops, efficient version of the previous. """
    fixed_edges, fixed_nodes = [], []

    def check_path_IP(G, ds, dd):
        SG = get_supply_graph(G)
        path, _, _, is_working = mxv.protocol_routing_IP(SG, ds, dd)
        return is_working

    if path_to_fix is not None:

        ds, dd = path_to_fix[0], path_to_fix[-1]

        path_elements = [path_to_fix[0]]  # nodes alternated with edges
        for i in range(len(path_to_fix) - 1):
            n1, n2 = path_to_fix[i], path_to_fix[i+1]
            path_elements.append((n1, n2))
            path_elements.append(n2)

        for e in path_elements:
            if type(e) is int:
                did_repair = do_repair_node(G, e)
                if did_repair:
                    fixed_nodes.append(e)

                if check_path_IP(G, ds, dd):
                    return fixed_nodes, fixed_edges

            elif type(e) is tuple:
                n1, n2 = make_existing_edge(e[0], e[1])
                did_repair = do_repair_edge(G, n1, n2)
                if did_repair:  # REPAIR EDGE
                    fixed_edges.append((n1, n2))

                if check_path_IP(G, ds, dd):  # CHECK PATH
                    return fixed_nodes, fixed_edges

    return fixed_nodes, fixed_edges


def do_repair_node(G, n):
    """ counts the repairs! """
    did_repair = G.nodes[n][co.ElemAttr.STATE_TRUTH.value] == co.NodeState.BROKEN.value
    # did_repair = G.nodes[n][co.ElemAttr.POSTERIOR_BROKEN.value] > 0

    G.nodes[n][co.ElemAttr.STATE_TRUTH.value] = co.NodeState.WORKING.value
    G.nodes[n][co.ElemAttr.POSTERIOR_BROKEN.value] = co.NodeState.WORKING.value  # discover
    print("repairing", n, did_repair)
    return n if did_repair else None


def do_repair_edge(G, n1, n2):
    """ counts the repairs! """
    did_repair = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value] == co.NodeState.BROKEN.value
    # did_repair = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value] > 0

    G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value] = co.NodeState.WORKING.value
    G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value] = co.NodeState.WORKING.value  # discover
    print("repairing", n1, n2, did_repair)
    return (n1, n2) if did_repair else None


def do_repair_full_edge(G, n1, n2):
    """ repairs all elements of the edge"""
    d0 = do_repair_edge(G, n1, n2)
    d1 = do_repair_node(G, n1)
    d2 = do_repair_node(G, n2)
    repn, repe = [], []
    if d0:
        repe.append((n1, n2))
    if d1:
        repn.append(n1)
    if d2:
        repn.append(n2)
    return repn, repe


def discover_edge(G, n1, n2, p_broken):
    """  Sets the posterior probability according to the actual state of the element. """
    G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value] = p_broken
    # assert(G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value] == G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value])


def discover_node(G, n, p_broken):
    """  Sets the posterior probability according to the actual state of the element. """
    G.nodes[n][co.ElemAttr.POSTERIOR_BROKEN.value] = p_broken
    # assert(G.nodes[n][co.ElemAttr.POSTERIOR_BROKEN.value] == G.nodes[n][co.ElemAttr.STATE_TRUTH.value])


def discover_path(G, path, p_broken):
    """  Sets the posterior probability according to the actual state of the path. """
    for i in range(len(path)-1):
        n1, n2 = make_existing_edge(path[i], path[i + 1])
        G.nodes[n1][co.ElemAttr.POSTERIOR_BROKEN.value] = p_broken
        G.nodes[n2][co.ElemAttr.POSTERIOR_BROKEN.value] = p_broken
        G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value] = p_broken
        assert (G.nodes[n1][co.ElemAttr.POSTERIOR_BROKEN.value] == G.nodes[n1][co.ElemAttr.STATE_TRUTH.value])
        assert (G.nodes[n2][co.ElemAttr.POSTERIOR_BROKEN.value] == G.nodes[n2][co.ElemAttr.STATE_TRUTH.value])
        assert(G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value] == G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value])


def discover_path_truth_limit_broken(G, path):
    """  Sets the posterior probability according to the actual state of the path. """
    for i in range(len(path)-1):
        n1, n2 = make_existing_edge(path[i], path[i + 1])
        ed_truth = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value]

        G.nodes[n1][co.ElemAttr.POSTERIOR_BROKEN.value] = G.nodes[n1][co.ElemAttr.STATE_TRUTH.value]
        G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value] = ed_truth
        if ed_truth == 1:
            break

        no_truth = G.nodes[n2][co.ElemAttr.STATE_TRUTH.value]
        G.nodes[n2][co.ElemAttr.POSTERIOR_BROKEN.value] = no_truth
        if no_truth == 1:
            break

        assert (G.nodes[n1][co.ElemAttr.POSTERIOR_BROKEN.value] == G.nodes[n1][co.ElemAttr.STATE_TRUTH.value])
        assert (G.nodes[n2][co.ElemAttr.POSTERIOR_BROKEN.value] == G.nodes[n2][co.ElemAttr.STATE_TRUTH.value])
        assert (G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value] == G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value])


# def discover_path(G, path, p_broken):
#     """  Sets the posterior probability according to the actual state of the path. """
#     for i in range(len(path)-1):
#         n1, n2 = make_existing_edge(G, path[i], path[i+1])
#         G.nodes[n1][co.ElemAttr.POSTERIOR_BROKEN.value] = p_broken
#         G.nodes[n2][co.ElemAttr.POSTERIOR_BROKEN.value] = p_broken
#         G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value] = p_broken
#         assert (G.nodes[n1][co.ElemAttr.POSTERIOR_BROKEN.value] == G.nodes[n1][co.ElemAttr.STATE_TRUTH.value])
#         assert (G.nodes[n2][co.ElemAttr.POSTERIOR_BROKEN.value] == G.nodes[n2][co.ElemAttr.STATE_TRUTH.value])
#         assert(G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value] == G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value])


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
        # n1, n2 = make_existing_edge(G, n1, n2)
        cap = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value]  # TODO
        capacities.append(cap)
    min_cap = min(capacities)
    return min_cap


def get_residual_demand(G):
    return sum([d for _, _, d in get_demand_edges(G, is_check_unsatisfied=True, is_capacity=True)])


def demand_pruning(G, path, quantity):
    """ Prune the edges capacity by quantity, assumed to be small enough. """
    d1, d2 = path[0], path[-1]  # make_existing_edge(G, path[0], path[-1])
    demand_full_capacity = G.edges[d1, d2, co.EdgeType.DEMAND.value][co.ElemAttr.CAPACITY.value]
    G.edges[d1, d2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value] -= quantity

    for i in range(len(path) - 1):
        n1, n2 = path[i], path[i + 1]
        # n1, n2 = make_existing_edge(G, n1, n2)
        full_cap = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.CAPACITY.value]
        cap =      G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value]
        assert(cap-quantity >= 0)
        G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value] -= quantity

        # debugging
        G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.SAT_DEM.value][(d1, d2)] += round(quantity/full_cap, 3)  # (demand edge): percentage flow
        G.edges[d1, d2, co.EdgeType.DEMAND.value][co.ElemAttr.SAT_SUP.value][(n1, n2)] += round(quantity/demand_full_capacity, 3)  # (demand edge): percentage flow


def reset_supply_edges(G):
    """ Resets the capacity of the suply edges of a graph to the original values. Also demand edges are reset """
    for d1, d2, _ in get_demand_edges(G):
        original_cap = G.edges[d1, d2, co.EdgeType.DEMAND.value][co.ElemAttr.CAPACITY.value]
        G.edges[d1, d2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value] = original_cap

    for n1, n2, ty in G.edges:
        if ty == co.EdgeType.SUPPLY.value:
            original_cap = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.CAPACITY.value]
            G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value] = original_cap


def demand_node_position(demand_edges, demands, nodes):
    """ Maps every node to their position in the graph wrt the demand graph 0 source, 1 mid, 2 target. """
    demand_node_pos = {}

    for node in nodes:
        for h in demands:
            demand_node_pos[node, h] = 1  # 1 nodo centrale di demand

    sources = [i for i, _, _ in demand_edges]  # [1,2,3]
    targets = [j for _, j, _ in demand_edges]  # [45,67,88]

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


def get_path_elements(path):
    """ returns the set of nodes and edges in path"""
    nodes, edges = set(), set()
    for i in range(len(path)-1):
        n1, n2 = path[i], path[i+1]
        nodes |= {n1, n2}
        edges |= {(n1, n2)}
    return nodes, edges


def get_path_cost_VN(G, path_nodes, is_oracle=False):
    """ VERSIONE NUOVA: returns the expected repair cost. """
    cap = get_path_residual_capacity(G, path_nodes)

    if cap == 0:
        return np.inf

    cap = min(cap, get_residual_demand(G))

    cost_broken_els_exp = 0
    attribute = co.ElemAttr.STATE_TRUTH.value if is_oracle else co.ElemAttr.POSTERIOR_BROKEN.value

    # expected cost of repairing the nodes
    for n1 in path_nodes:
        posterior_broken_node = G.nodes[n1][attribute]
        cost_broken_els_exp += co.REPAIR_COST * posterior_broken_node
        # print(G.nodes[n1][co.ElemAttr.STATE_TRUTH.value], posterior_broken_node, n1, cost_broken_els_exp)

    # expected cost of repairing the edges
    for i in range(len(path_nodes) - 1):
        n1, n2 = path_nodes[i], path_nodes[i + 1]
        n1, n2 = make_existing_edge(n1, n2)
        posterior_broken_edge = G.edges[n1, n2, co.EdgeType.SUPPLY.value][attribute]
        cost_broken_els_exp += co.REPAIR_COST * posterior_broken_edge / cap
        # print(G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value], posterior_broken_edge, n1, n2, cost_broken_els_exp)

    # exp_inutility = cost_broken_els_exp   # [d1, d2, d3] d1=(n1, n2) -- [(p1, m1), p2, p3] -> arg min
    # exp_inutility = exp_inutility + (np.inf if cap == 0 else 0)
    # print("hey", cost_broken_els + cost_broken_els_exp, cap, exp_cost)
    return cost_broken_els_exp


def get_path_cost_cedarlike(G, path_nodes):
    cap = get_path_residual_capacity(G, path_nodes)

    if cap == 0:
        return np.inf

    cost_broken_els_exp = 0

    # print(path_nodes)
    # expected cost of repairing the nodes
    for n1 in path_nodes:
        is_unk_bro = int(G.nodes[n1][co.ElemAttr.POSTERIOR_BROKEN.value] > 0)
        cost_broken_els_exp += co.REPAIR_COST * is_unk_bro
        # print(G.nodes[n1][co.ElemAttr.STATE_TRUTH.value], posterior_broken_node, n1, cost_broken_els_exp)

    # expected cost of repairing the edges
    for i in range(len(path_nodes) - 1):
        n1, n2 = path_nodes[i], path_nodes[i + 1]
        n1, n2 = make_existing_edge(n1, n2)
        is_unk_bro = int(G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value] > 0)
        cost_broken_els_exp += co.REPAIR_COST * is_unk_bro / cap + (1-is_unk_bro) / cap

    return cost_broken_els_exp


def probabilistic_edge_weights(SG, G):
    for n1, n2, et in SG.edges:
        # n1, n2 = make_existing_edge(G, n1, n2)

        edge_cap = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value]
        edge_known = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value]
        broken_edge = co.REPAIR_COST * edge_known

        n1_known = G.nodes[n1][co.ElemAttr.POSTERIOR_BROKEN.value]
        broken_n1 = co.REPAIR_COST * n1_known

        n2_known = G.nodes[n2][co.ElemAttr.POSTERIOR_BROKEN.value]
        broken_n2 = co.REPAIR_COST * n2_known

        # weight = np.inf if edge_cap == 0 else (broken_edge + (broken_n1 + broken_n2) / 2) * (1/(edge_cap + co.epsilon) + 1)  # 1 stands for the crossing
        weight = np.inf if edge_cap == 0 else (broken_edge + (broken_n1 + broken_n2) / 2) #* (1/(edge_cap + co.epsilon))  # 1 stands for the crossing

        SG.edges[n1, n2, et][co.ElemAttr.WEIGHT.value] = weight
        G.edges[n1, n2, et][co.ElemAttr.WEIGHT.value] = weight


def broken_elements_in_path_T(G, path_nodes):
    """ Returns the broken elements in path TRUTH. """
    n_broken = 0
    for i in range(len(path_nodes) - 1):
        n1, n2 = path_nodes[i], path_nodes[i + 1]
        n1, n2 = make_existing_edge(n1, n2)

        if G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value] == 1:
            n_broken += 1
        if G.nodes[n1][co.ElemAttr.STATE_TRUTH.value] == 1:
            n_broken += 1

    if G.nodes[n2][co.ElemAttr.STATE_TRUTH.value] == 1:
        n_broken += 1
    return n_broken


def set_infinite_weights(G):
    """ Sets to infinity all the weights of the edges associated to path. """
    for n1, n2, et in get_supply_edges(G):
        if G.edges[n1, n2, et][co.ElemAttr.RESIDUAL_CAPACITY.value] == 0:
            G.edges[n1, n2, et][co.ElemAttr.WEIGHT.value] = np.inf
            G.edges[n1, n2, et][co.ElemAttr.WEIGHT_UNIT.value] = np.inf


def saturating_paths(G, d_edges):
    """ For all demands D, it returns all the paths that saturate D along with their capacity. """
    demands = {}
    for d1, d2, c in d_edges:
        res_capacity = c
        GCO = G.copy()
        SG = get_supply_graph(GCO)
        all_paths_saturate, capacity_all_paths_saturate = [], []

        while res_capacity > 0:  # reset residual capacity adding paths
            path, _, _ = mxv.protocol_repair_cedarlike(SG, d1, d2)
            quantity = do_prune(GCO, path, is_log=False)

            if quantity > 0:  # this is not ok
                res_capacity -= quantity
                all_paths_saturate.append(path)
                capacity_all_paths_saturate.append(quantity)
            else:
                break

        demands[(d1, d2)] = all_paths_saturate, capacity_all_paths_saturate
    return demands


def best_centrality_node(G):
    """ CEDAR centrality node. """
    nodes_centrality = []
    d_edges = get_demand_edges(G, is_check_unsatisfied=True, is_capacity=True)
    candidates = list(set(G.nodes) - set(get_monitor_nodes(G)))
    saturating_paths_vec = saturating_paths(G, d_edges)

    for n in tqdm.tqdm(candidates):
        centrality = demand_based_centrality(G, n, d_edges, saturating_paths_vec)
        nodes_centrality.append(centrality)

    bc = candidates[np.argmax(nodes_centrality)]
    print("Highest centrality node is", bc)  # TODO: centrality node can be the same
    return bc


def demand_based_centrality(G, n, d_edges, saturating_paths_vec):
    """ Computes the centrality to find the best centrality node. """

    centrality = 0
    for d1, d2, c in d_edges:
        all_paths_saturate, capacity_all_paths_saturate = saturating_paths_vec[(d1, d2)]
        capacity_paths_saturate_through_n = []
        for i, path in enumerate(all_paths_saturate):
            path_capacity = capacity_all_paths_saturate[i] if n in path else 0
            capacity_paths_saturate_through_n.append(path_capacity)

        ratio = sum(capacity_paths_saturate_through_n) / sum(capacity_all_paths_saturate)
        centrality += ratio * c
    return centrality
