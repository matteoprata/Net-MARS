
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal
import random
from src.utilities import util

import src.constants as co
from src.constants import EdgeType
from collections import defaultdict

# 1 -- complete_destruction
def complete_destruction(graph):
    """ destroys all the graph. """
    do_break_graph_components(graph, graph.nodes, graph.edges)
    return None, graph.nodes, graph.edges


# 2 -- uniform_destruction
def uniform_destruction(graph, ratio=.5):
    """ destroys random uniform components of the graph. """
    n_broken_nodes, n_broken_edges = int(len(graph.nodes) * ratio), int(len(graph.edges) * ratio)
    broken_nodes = random.sample(graph.nodes, n_broken_nodes)

    broken_edges = random.sample(graph.edges, n_broken_edges)
    do_break_graph_components(graph, broken_nodes, broken_edges)
    return None, broken_nodes, broken_edges


# 3 -- gaussian_destruction
def gaussian_destruction(graph, density, dims_ratio, destruction_width, n_disruption):

    x_density = round(dims_ratio["x"]*density)
    y_density = round(dims_ratio["y"]*density)

    def get_distribution():
        """ Destroys random gaussian components of the graph. """

        x = np.linspace(0, x_density/density, x_density)
        y = np.linspace(0, y_density/density, y_density)

        X, Y = np.meshgrid(x, y)
        pos = np.empty(X.shape + (2,))

        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        rvs = []
        # random variables of the epicenter
        for it in range(n_disruption):
            coo_mu  = [np.random.rand(1, 1)[0][0]*x_density/density, np.random.rand(1, 1)[0][0]*y_density/density]
            coo_var = [np.random.rand(1, 1)[0][0]*x_density/density, np.random.rand(1, 1)[0][0]*y_density/density]

            rv = multivariate_normal([coo_mu[0], coo_mu[1]], [[destruction_width*coo_var[0], 0], [0, destruction_width*coo_var[1]]])
            rvs.append(rv)

        # maximum of the probabilities, to merge epicenters
        distribution = np.maximum(rvs[0].pdf(pos), rvs[1].pdf(pos))
        for ir in range(2, len(rvs)):
            distribution = np.maximum(distribution, rvs[ir].pdf(pos))

        # plot3Ddisruption(X, Y, distribution)
        return distribution

    def plot3Ddisruption(X, Y, distribution):
        """ Plot the disaster is 3D. """
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, distribution, cmap='viridis', linewidth=0)

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.show()

    def sample_broken_element(list_broken, element, dist_max, dist, x, y):
        """ Break the element with probability given by the probability density function. """
        prob = util.min_max_normalizer(dist[x, y], 0, dist_max, 0, 1)
        state = np.random.choice(["BROKEN", "WORKING"], 1, p=[prob, 1 - prob])  # broken, working
        if state == "BROKEN":
            list_broken.append(element)

    distribution = get_distribution()
    distribution = np.flip(distribution, axis=0)  # coordinates systems != matrix system
    dist_max = np.max(distribution)

    broken_nodes, broken_edges = [], []

    # break edges probabilistically
    for n1 in graph.nodes:
        x, y = graph.nodes[n1][co.ElemAttr.LONGITUDE.value], graph.nodes[n1][co.ElemAttr.LATITUDE.value]
        y, x = graph_coo_to_grid(x, y, density, x_density, y_density) # swap rows by columns notation, array index by rows (y)
        sample_broken_element(broken_nodes, n1, dist_max, distribution, x, y)

    #break edges probabilistically
    for edge in graph.edges:
        n1, n2, _ = edge
        x0, y0 = graph.nodes[n1][co.ElemAttr.LONGITUDE.value], graph.nodes[n1][co.ElemAttr.LATITUDE.value]
        x1, y1 = graph.nodes[n2][co.ElemAttr.LONGITUDE.value], graph.nodes[n2][co.ElemAttr.LATITUDE.value]
        x, y = (x0+x1)/2, (y0+y1)/2   # break edge from it's midpoint for simplicity
        y, x = graph_coo_to_grid(x, y, density, x_density, y_density)
        sample_broken_element(broken_edges, edge, dist_max, distribution, x, y)

    do_break_graph_components(graph, broken_nodes, broken_edges)
    return distribution, broken_nodes, broken_edges


def gaussian_progressive_destruction(graph, density, dims_ratio, destruction_quantity, config, n_bins=50, mu=0, sig=.5, distance_factor=1):
    x_density = round(dims_ratio["x"]*density)
    y_density = round(dims_ratio["y"]*density)

    # TODO: check randomness
    # the epicenter is a randomly picked node
    if not config.is_xindvar_destruction:
        np.random.seed(config.fixed_unvarying_seed)

    epicenter = random.choice(graph.nodes)['id']

    if not config.is_xindvar_destruction:
        np.random.seed(config.seed)

    graph.nodes[epicenter][co.ElemAttr.IS_EPICENTER.value] = 1

    x, y = graph.nodes[epicenter][co.ElemAttr.LONGITUDE.value], graph.nodes[epicenter][co.ElemAttr.LATITUDE.value]
    epiy, epix = graph_coo_to_grid(x, y, density, x_density, y_density)
    epicenter = np.array([epix, epiy])

    # list the distance of every node to the epicenter
    node_distances = []
    for n1 in graph.nodes:
        x, y = graph.nodes[n1][co.ElemAttr.LONGITUDE.value], graph.nodes[n1][co.ElemAttr.LATITUDE.value]
        y, x = graph_coo_to_grid(x, y, density, x_density, y_density)
        node_pos = np.asarray([x, y])
        dist = np.linalg.norm(node_pos - epicenter)
        node_distances.append(dist)

    print(list(zip(graph.nodes, node_distances)))

    # list the distance of every edge to the epicenter
    edge_distances = []
    for n1, n2, _ in graph.edges:
        x0, y0 = graph.nodes[n1][co.ElemAttr.LONGITUDE.value], graph.nodes[n1][co.ElemAttr.LATITUDE.value]
        x1, y1 = graph.nodes[n2][co.ElemAttr.LONGITUDE.value], graph.nodes[n2][co.ElemAttr.LATITUDE.value]
        x, y = (x0+x1)/2, (y0+y1)/2   # break edge from it's midpoint for simplicity
        y, x = graph_coo_to_grid(x, y, density, x_density, y_density)
        node_pos = np.asarray([x, y])
        dist = np.linalg.norm(node_pos - epicenter)
        edge_distances.append(dist)

    elements_list = list(graph.nodes) + list(graph.edges)
    elements_distances = node_distances + edge_distances
    n_elements = len(elements_list)
    max_distance = max(elements_distances) * (1+ distance_factor)

    bins = [i * (max_distance / n_bins) for i in range(1, n_bins+1)]
    inds = np.digitize(elements_distances, bins)  # say to what bin each distance belongs

    bins_dict = defaultdict(list)
    el_bin_dict = dict()
    for i, ind in enumerate(inds):
        bins_dict[ind].append(elements_list[i])  # map the bin id to the elements in that bin
        el_bin_dict[elements_list[i]] = ind

    for k in range(len(bins)):
        if k not in bins_dict.keys():
            bins_dict[k] = []  # for the empty bins

    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    bins_1 = np.array([i * (1 / n_bins) for i in range(n_bins+1)])
    bins_2 = np.roll(np.asarray(bins_1), -1)
    meanw = ((bins_2 + bins_1) / 2)[:-1]
    probs = gaussian(meanw, sig=sig, mu=mu)  # probability of broken elements in each bin

    # dt = pd.DataFrame({"probs": probs})
    # li_bi = [len(bins_dict[k]) for k in bins_dict]
    # dt["cou"] = np.array(li_bi) / max(li_bi)
    # dt["probs"].plot()
    # plt.bar(x=dt.index, height=dt["cou"])
    # plt.show()

    broken = []
    for slice in range(n_bins):
        perc_broken_sofar = (len(broken) / n_elements)
        n_expected_broken_slice = len(bins_dict[slice]) * probs[slice]
        perc_expected_broken_sofar = n_expected_broken_slice / n_elements + perc_broken_sofar

        if perc_expected_broken_sofar > destruction_quantity:              # I will break more than threshold
            perc_to_break = abs(perc_broken_sofar - destruction_quantity)  # with respect to the total number of elements
            n_expected_broken_slice = min(n_elements * perc_to_break, len(bins_dict[slice]))
            broken += random.sample(bins_dict[slice], int(np.ceil(n_expected_broken_slice)))
            break
        else:
            broken += random.sample(bins_dict[slice], int(np.ceil(n_expected_broken_slice)))

    perc_broken_sofar = (len(broken) / n_elements)
    assert(util.is_distance_tolerated(perc_broken_sofar, destruction_quantity, .05),
           "Percentage of disruption is insufficient.")

    broken_nodes, broken_edges = [], []
    for el in broken:
        if type(el) is tuple:
            broken_edges.append(el)
        else:
            broken_nodes.append(el)

    do_break_graph_components(graph, broken_nodes, broken_edges)
    return None, broken_nodes, broken_edges


# DESTROY GRAPH
def do_break_graph_components(graph, broken_nodes, broken_edges):
    for n1 in broken_nodes:
        destroy_node(graph, n1)

    for n1, n2, _ in broken_edges:
        destroy_edge(graph, n1, n2)


def destroy_node(graph, node_id):
    graph.nodes[node_id][co.ElemAttr.STATE_TRUTH.value] = co.NodeState.BROKEN.value


def destroy_edge(graph, node_id_1, node_id_2):
    graph.edges[node_id_1, node_id_2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value] = co.NodeState.BROKEN.value


def graph_coo_to_grid(x, y, density, x_density, y_density):
    """ Given [0,1] coordinates, it returns the coordinates of the relative [0, density] coordinates. """
    xn = min(round(x*density), x_density-1)
    yn = min(round(y*density), y_density-1)
    return xn, yn