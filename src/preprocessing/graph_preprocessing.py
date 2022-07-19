
import random
import numpy as np
import networkx as nx
import os
import pandas as pd

import src.plotting.graph_plotting as gp
import src.preprocessing.graph_distruction as dis
import src.preprocessing.graph_utils as grau

import src.constants as co
import src.utilities.util as util
from collections import defaultdict
from matplotlib import pyplot as plt
from itertools import combinations
from filelock import FileLock


# 1 -- init graph G
def init_graph(path_to_graph, graph_name, supply_capacity, config):
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
                                co.ElemAttr.WEIGHT_UNIT.value: 1,
                                co.ElemAttr.PRIOR_BROKEN.value: config.UNK_prior,
                                co.ElemAttr.POSTERIOR_BROKEN.value: config.UNK_prior,
                                co.ElemAttr.ID.value: element_id,
                                })])

    # every edge will work by default
    for n1, n2, gt in raw_graph.edges:
        if not (n1 in ignore_nodes or n2 in ignore_nodes):
            element_id += 1
            elements_val_id[(n1, n2)] = element_id
            elements_val_id[(n2, n1)] = element_id
            elements_id_val[element_id] = (n1, n2)
            supply_capacity_rand = config.rand_generator_capacities.randint(*config.supply_capacity)
            G.add_edges_from([(n1, n2, co.EdgeType.SUPPLY.value, {co.ElemAttr.STATE_TRUTH.value: co.NodeState.WORKING.value,  # unobservable state
                                                                  co.ElemAttr.CAPACITY.value: supply_capacity_rand,
                                                                  co.ElemAttr.RESIDUAL_CAPACITY.value: supply_capacity_rand,
                                                                  co.ElemAttr.WEIGHT.value: -1,
                                                                  co.ElemAttr.WEIGHT_UNIT.value: 1,
                                                                  co.ElemAttr.PRIOR_BROKEN.value: config.UNK_prior,
                                                                  co.ElemAttr.POSTERIOR_BROKEN.value: config.UNK_prior,
                                                                  co.ElemAttr.ID.value: element_id,
                                                                  co.ElemAttr.SAT_DEM.value: defaultdict(int)
                                                                  })])

    # ADD the backbones!
    if config.graph_dataset == co.GraphName.MINNESOTA and config.is_minnesota_backbone_on:
        place_static_backbone(G, co.MINNESOTA_STP_BACKBONE, config.backbone_capacity)

    # ADD probability RESISTANCE_TO_DESTRUCTION:
    if config.algo_name == co.Algorithm.TOMO_CEDAR_DYN:
        for n in G.nodes:
            G.nodes[n][co.ElemAttr.RESISTANCE_TO_DESTRUCTION.value] = config.uniform_resistance_destruction

        for n1, n2, ty in G.edges:
            if ty == co.EdgeType.SUPPLY:
                G.edges[n1, n2, ty][co.ElemAttr.RESISTANCE_TO_DESTRUCTION.value] = config.uniform_resistance_destruction

    return G, elements_val_id, elements_id_val


def place_static_backbone(G, path_edges, backbone_capacity):
    SG = grau.get_supply_graph(G)
    for na, nb in path_edges:
        path = nx.shortest_path(SG, na, nb)
        for i in range(len(path)-1):
            ea, eb = grau.make_existing_edge(G, path[i], path[i+1])
            G.edges[ea, eb, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value] = backbone_capacity
            G.edges[ea, eb, co.EdgeType.SUPPLY.value][co.ElemAttr.CAPACITY.value] = backbone_capacity


def place_backbone(G, config):
    max_comp = list(get_max_component(G))

    np.random.seed(config.fixed_unvarying_seed)  # this does not vary
    list_pairs = [np.random.choice(max_comp, size=2, replace=False) for _ in range(config.n_backbone_pairs)]
    np.random.seed(config.seed)

    for p1, p2 in list_pairs:
        path = nx.shortest_path(G, p1, p2)
        for i in range(len(path) - 1):
            e1, e2 = path[i], path[i + 1]
            e1, e2 = grau.make_existing_edge(G, e1, e2)
            backbone_flow = config.supply_capacity[0] * (1 + config.percentage_flow_backbone)
            G.edges[e1, e2, co.EdgeType.SUPPLY.value][co.ElemAttr.CAPACITY.value] = backbone_flow
            G.edges[e1, e2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value] = backbone_flow

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
def destroy(G, destruction_type, destruction_precision, dims_ratio, destruction_width, n_destruction, graph_name, sim_seed, config, ratio=None):
    """ Handles three type of destruction. """

    if destruction_type == co.Destruction.GAUSSIAN:
        dist, nodes, edges = dis.gaussian_destruction(G, destruction_precision, dims_ratio, destruction_width, n_destruction)
    elif destruction_type == co.Destruction.GAUSSIAN_PROGRESSIVE:
        dist, nodes, edges = dis.gaussian_progressive_destruction(G, destruction_precision, dims_ratio, ratio, config=config)
    elif destruction_type == co.Destruction.UNIFORM:
        dist, nodes, edges = dis.uniform_destruction(G, ratio)
    elif destruction_type == co.Destruction.COMPLETE:
        dist, nodes, edges = dis.complete_destruction(G)
    else:
        dist, nodes, edges = None, None, None
        print("Unhandled destruction type.")
        exit()

    perc_broken_elements = (len(nodes) + len(edges)) / (len(G.nodes) + len(G.edges))
    print("percentage of broken elements", perc_broken_elements)

    #"data/porting/demand-s|{}-g|{}.csv"
    # write graph destruction for porting TODO: remove
    # dfn = pd.DataFrame(nodes)
    # dfe = pd.DataFrame(edges)
    # dfn.to_csv("data/porting/destruction-s|{}-g|{}-des|{}-n.csv".format(sim_seed, graph_name.name, destruction_type.value))
    # dfe.to_csv("data/porting/destruction-s|{}-g|{}-des|{}-e.csv".format(sim_seed, graph_name.name, destruction_type.value))
    return dist, nodes, edges, perc_broken_elements


# 4 -- print_graph_info G
def print_graph_info(G):
    print("graph has nodes:", len(G.nodes), "and edges:", len(G.edges))


def select_demand(G, max_comp, n_samples, is_nodes, THRESHOLD_DIST=0.7):
    """ This produces demand nodes or pairs on the frontier of the graph."""
    # # CEREFUL n^2 complexity
    nodes_dis = dict()
    for ns in max_comp:
        for nt in max_comp:
            x1, y1 = G.nodes[ns][co.ElemAttr.LONGITUDE.value], G.nodes[ns][co.ElemAttr.LATITUDE.value]
            x2, y2 = G.nodes[nt][co.ElemAttr.LONGITUDE.value], G.nodes[nt][co.ElemAttr.LATITUDE.value]
            dist = np.linalg.norm(np.asarray([x1, y1]) - np.asarray([x2, y2]))
            nodes_dis[(ns, nt)] = dist

    list_pairs_dist = sorted(nodes_dis.items(), key=lambda x: x[1], reverse=True)  # [(n1, n2), dis] # sorted pairs decreasing orders

    distances = np.array([B[1] for B in list_pairs_dist])
    pairs = np.array([B[0] for B in list_pairs_dist])
    mask_ok_distance = distances >= THRESHOLD_DIST  # [(1,2), (2,3)]  [1,2,3]

    if is_nodes:  # returns nodes
        pairs_to_sample = np.array(list(set(pairs[mask_ok_distance].flatten())))
        np.random.shuffle(pairs_to_sample)
        list_pairs = pairs_to_sample[:n_samples]
    #
    else:  # returns pairs of nodes
        pairs_to_sample = pairs[mask_ok_distance]
        np.random.shuffle(pairs_to_sample)
        list_pairs = pairs_to_sample[:n_samples, :]

    return list_pairs


# 6 -- demand pairs
def add_demand_pairs(G, n_demand_pairs, demand_capacity, config):
    assert(False)  # because we want to add demand edges progressively with seeds
    max_comp = list(get_max_component(G))

    # if we are varying the destruction probability the demand edges need to stay same
    # if config.experiment_ind_var == co.IndependentVariable.PROB_BROKEN:
    #     np.random.seed(config.fixed_unvarying_seed)  # destruction varies > vary only the epicenter

    list_pairs = select_demand(G, max_comp, n_demand_pairs, False)

    # assert config.is_xindvar_destruction and config.n_demand_pairs <= 8 # otherwise, pick them at random!
    # list_pairs = [(60, 411), (360, 522), (186, 78), (27, 221), (79, 474), (397, 525), (83, 564), (373, 281)]
    # list_pairs = list_pairs[:n_demand_pairs]

    # set the old seed back again
    # if config.experiment_ind_var == co.IndependentVariable.PROB_BROKEN:
    #     np.random.seed(config.seed)

    demand_edges = set()
    demand_nodes = set()
    for demand_pair in list_pairs:
        n1, n2 = demand_pair[0], demand_pair[1]
        G.add_edge(n1, n2, co.EdgeType.DEMAND.value)
        G.edges[n1, n2, co.EdgeType.DEMAND.value][co.ElemAttr.STATE_TRUTH.value] = co.NodeState.NA.value
        G.edges[n1, n2, co.EdgeType.DEMAND.value][co.ElemAttr.CAPACITY.value] = demand_capacity
        G.edges[n1, n2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value] = demand_capacity
        G.edges[n1, n2, co.EdgeType.DEMAND.value][co.ElemAttr.SAT_SUP.value] = defaultdict(int)

        demand_edges.add((n1, n2))
        demand_nodes.add(n1)
        demand_nodes.add(n2)
    return demand_nodes, demand_edges


def get_max_component(G):
    """ Get the biggest connected component to sample the demand pairs from. """
    max_val, max_comp = 0, None
    for n in G.nodes:
        mates = nx.node_connected_component(G, n)
        n_mates = len(mates)
        if n_mates > max_val:
            max_val = n_mates
            max_comp = mates
    return max_comp


def add_demand_clique(G, n_demand_nodes, demand_capacity, config):
    """ Add edges as a clique. """
    max_comp = list(get_max_component(G))  # biggest connected component in G

    # pick higher degree nodes with higher probability
    # degs = G.degree()
    # max_comp_degs = np.array([degs[n] for n in max_comp])
    # max_comp_degs = max_comp_degs / np.sum(max_comp_degs)
    # is_biased = False

    # choose n_demand_nodes randomly in the graph
    if config.experiment_ind_var == co.IndependentVariable.N_DEMAND_EDGES:
        np.random.seed(config.fixed_unvarying_seed)  # do not vary the positions of the demands

    # ----> FIXING DEMAND NODES and EDGES
    nv_index = None  # read from file the edges in the graph
    if os.path.exists(config.edges_list_path):
        while not os.access(config.edges_list_path, os.R_OK):
            pass
        dict_edges_r = util.read_pickle(config.edges_list_path)

        if config.seed in dict_edges_r.keys():
            dict_edges_s = dict_edges_r[config.seed]
            nv_index = util.nearest_value_index(value=config.n_demand_clique, list_values=list(dict_edges_s.keys()))

        dict_edges = defaultdict(lambda: defaultdict(set))
        for k in dict_edges_r:
            dict_edges[k] = dict_edges_r[k]
    else:
        dict_edges = defaultdict(lambda: defaultdict(set))

    prev_edges = dict_edges[config.seed][nv_index] if nv_index is not None else set()
    print("Previous edges", prev_edges)

    list_nodes_tot = select_demand(G, max_comp, config.n_demand_clique, is_nodes=True)  # ndn demand nodes
    list_nodes = list_nodes_tot[:n_demand_nodes]  # all clique nodes

    clique_edges = np.asarray(list(combinations(list_nodes, r=2)))                       # all clique edges
    clique_edges_available = list(set([tuple(se) for se in clique_edges]) - prev_edges)  # removes edges used before
    n_new_edges = config.n_demand_pairs - len(prev_edges)                                # how many new edges to add
    clique_edges_idxs = np.random.choice(np.arange(len(clique_edges_available)), size=n_new_edges, replace=False)
    new_edges = {clique_edges_available[ind] for ind in clique_edges_idxs}
    # print(prev_edges, new_edges)  # new_edges is empty if before it was set and store on file

    total_edges = prev_edges | new_edges
    dict_edges[config.seed][config.n_demand_clique] = total_edges
    print(config.seed, dict_edges)

    # needed to serialize...
    ser_dict = dict({int(k): dict({int(k1): set((int(li[0]), int(li[1])) for li in dict_edges[k][k1]) for k1 in dict_edges[k]}) for k in dict_edges})

    while os.path.exists(config.edges_list_path) and not os.access(config.edges_list_path, os.W_OK):
        print("waiting")
        pass
    util.write_pickle(ser_dict, config.edges_list_path)

    if config.experiment_ind_var == co.IndependentVariable.N_DEMAND_EDGES:
        np.random.seed(config.seed)

    # print("\nDegrees")
    demand_edges = set()
    demand_nodes = set()
    for i, edge in enumerate(total_edges):
        n1, n2 = edge
        G.add_edge(n1, n2, co.EdgeType.DEMAND.value)
        G.edges[n1, n2, co.EdgeType.DEMAND.value][co.ElemAttr.STATE_TRUTH.value] = co.NodeState.NA.value
        G.edges[n1, n2, co.EdgeType.DEMAND.value][co.ElemAttr.CAPACITY.value] = demand_capacity
        G.edges[n1, n2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value] = demand_capacity
        G.edges[n1, n2, co.EdgeType.DEMAND.value][co.ElemAttr.SAT_SUP.value] = defaultdict(int)

        demand_edges.add((n1, n2))
        demand_nodes.add(n1)
        demand_nodes.add(n2)

    return demand_nodes, demand_edges
