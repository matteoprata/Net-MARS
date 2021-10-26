
import networkx as nx

import src.preprocessing.graph_preprocessing as gp
from src.preprocessing.graph_preprocessing import *
import src.constants as co


def run(config):

    # read graph and print stats
    G = init_graph(co.path_to_graph, config.graph_name, config.supply_capacity)
    print_graph_info(G)

    # normalize coordinates and break components
    dim_ratio = scale_coordinates(G)
    distribution, broken_nodes, broken_edges = destroy(G, config.destruction_type, config.destruction_precision, dim_ratio, config.destruction_width, config.n_destruction)

    # add_demand_endpoints
    D = add_demand_pairs(G, config.n_demand_pairs, config.demand_capacity)

    # plot graph
    plot_graph(G, config.graph_name, distribution, config.destruction_precision, dim_ratio, config.destruction_show_plot, config.destruction_save_plot, config.seed)
