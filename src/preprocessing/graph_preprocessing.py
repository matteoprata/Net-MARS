
import numpy as np
import networkx as nx
import os

import src.plotting.graph_plotting as gp
import src.preprocessing.graph_distruction as dis
import src.constants as co
import src.utilities.util as util


# 1 -- init graph G
def init_graph(path_to_graph, graph_name, supply_capacity):
    """
    Set the labels for the graph components.
    Inputs graph must maintain the information of the node (ID, longitude x, latitude y, state) for the edges (capacity, state).
    """
    def load_graph(path_to_graph, graph_name):
        return nx.MultiGraph(nx.read_gml(os.path.join(path_to_graph, graph_name), label='id'))

    raw_graph = load_graph(path_to_graph, graph_name)
    G = nx.MultiGraph()

    elements_id_val = {}
    elements_val_id = {}

    ignore_nodes = []
    element_id = -1  # counter of the ids
    # every node will work by default
    for n1 in raw_graph.nodes:
        if not (co.ElemAttr.LATITUDE.value in raw_graph.nodes[n1] and co.ElemAttr.LONGITUDE.value in raw_graph.nodes[n1]):
            ignore_nodes.append(n1)
            continue

        element_id += 1
        elements_val_id[n1] = element_id
        elements_id_val[element_id] = n1

        G.add_nodes_from([(n1, {co.ElemAttr.STATE_TRUTH.value: co.NodeState.WORKING.value,  # unobservable state
                                co.ElemAttr.LATITUDE.value: float(raw_graph.nodes[n1][co.ElemAttr.LATITUDE.value]),
                                co.ElemAttr.LONGITUDE.value: float(raw_graph.nodes[n1][co.ElemAttr.LONGITUDE.value]),
                                co.ElemAttr.WEIGHT.value: -1,
                                co.ElemAttr.PRIOR_BROKEN.value: 0.5,
                                co.ElemAttr.POSTERIOR_BROKEN.value: 0.5,
                                co.ElemAttr.ID.value: element_id,
                                })])

    # every edge will work by default
    for n1, n2, gt in raw_graph.edges:
        if not (n1 in ignore_nodes or n2 in ignore_nodes):
            element_id += 1
            elements_val_id[(n1, n2)] = element_id
            elements_val_id[(n2, n1)] = element_id
            elements_id_val[element_id] = (n1, n2)

            G.add_edges_from([(n1, n2, co.EdgeType.SUPPLY.value, {co.ElemAttr.STATE_TRUTH.value: co.NodeState.WORKING.value,  # unobservable state
                                                                  co.ElemAttr.CAPACITY.value: supply_capacity,
                                                                  co.ElemAttr.RESIDUAL_CAPACITY.value: supply_capacity,
                                                                  co.ElemAttr.WEIGHT.value: -1,
                                                                  co.ElemAttr.PRIOR_BROKEN.value: 0.5,
                                                                  co.ElemAttr.POSTERIOR_BROKEN.value: 0.5,
                                                                  co.ElemAttr.ID.value: element_id,
                                                                  })])
    return G, elements_val_id, elements_id_val


# 2 -- scale graph G
def scale_coordinates(G):
    """ Scale graph coordinates to positive [0,1] """

    def get_dimensions():
        """ gets the max/min longitude/latitude and retursn it"""
        coo_lats = [G.nodes[n1][co.ElemAttr.LATITUDE.value] for n1 in G.nodes]
        coo_long = [G.nodes[n1][co.ElemAttr.LONGITUDE.value] for n1 in G.nodes]
        return max(coo_long), max(coo_lats), min(coo_long), min(coo_lats)

    max_long, max_lat, min_long, min_lat = get_dimensions()

    # maximum horizontal and vertical distance
    dis_xy = [np.linalg.norm(np.array([min_long, min_lat]) - np.array([max_long, min_lat])),
              np.linalg.norm(np.array([min_long, min_lat]) - np.array([min_long, max_lat]))]

    # shift on the positive quadrant and in [0,1]
    for n1 in G.nodes:
        G.nodes[n1][co.ElemAttr.LATITUDE.value] = util.min_max_normalizer(G.nodes[n1][co.ElemAttr.LATITUDE.value], min_lat, max_lat, 0, dis_xy[1]) / max(dis_xy)
        G.nodes[n1][co.ElemAttr.LONGITUDE.value] = util.min_max_normalizer(G.nodes[n1][co.ElemAttr.LONGITUDE.value], min_long, max_long, 0, dis_xy[0]) / max(dis_xy)

    max_long, max_lat, min_long, min_lat = get_dimensions()
    max_dist, min_dist = max(max_long, max_lat), min(max_long, max_lat)

    ratio = {"x": max_long/max_dist, "y": max_lat/max_dist}
    print("graph dims ratio:", ratio)
    return ratio


# 3 -- destroy graph G
def destroy(G, destruction_type, destruction_precision, dims_ratio, destruction_width, n_destruction, ratio=None):
    """ Handles three type of destruction. """

    if destruction_type == co.Destruction.GAUSSIAN:
        return dis.gaussian_destruction(G, destruction_precision, dims_ratio, destruction_width, n_destruction)
    elif destruction_type == co.Destruction.UNIFORM:
        return dis.uniform_destruction(G, ratio)
    elif destruction_type == co.Destruction.COMPLETE:
        return dis.complete_destruction(G)


# 4 -- print_graph_info G
def print_graph_info(G):
    print("graph has nodes:", len(G.nodes), "and edges:", len(G.edges))


# 6 -- demand pairs
def add_demand_pairs(G, n_demand_pairs, demand_capacity):
    list_pairs = [np.random.choice(G.nodes, size=2, replace=True) for _ in range(n_demand_pairs)]
    for demand_pair in list_pairs:
        n1, n2 = demand_pair[0], demand_pair[1]
        G.add_edge(n1, n2, co.EdgeType.DEMAND.value)
        G.edges[n1, n2, co.EdgeType.DEMAND.value][co.ElemAttr.STATE_TRUTH.value] = co.NodeState.NA.value
        G.edges[n1, n2, co.EdgeType.DEMAND.value][co.ElemAttr.CAPACITY.value] = demand_capacity
        G.edges[n1, n2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value] = demand_capacity



