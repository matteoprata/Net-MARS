
import networkx as nx

from src.preprocessing.graph_routability import *
from src.preprocessing.graph_preprocessing import *
from src.preprocessing.graph_monitoring import *
import src.constants as co


def run(config):

    # read graph and print stats
    G, elements_val_id, elements_id_val = init_graph(co.path_to_graph, config.graph_name, config.supply_capacity)
    print_graph_info(G)

    # normalize coordinates and break components
    dim_ratio = scale_coordinates(G)

    plot_graph(G, config.graph_name, None, config.destruction_precision, dim_ratio,
               config.destruction_show_plot, config.destruction_save_plot, config.seed, "ORI")

    distribution, broken_nodes, broken_edges = destroy(G, config.destruction_type, config.destruction_precision, dim_ratio, config.destruction_width, config.n_destruction)

    # add_demand_endpoints
    add_demand_pairs(G, config.n_demand_pairs, config.demand_capacity)

    # plot graph
    plot_graph(G, config.graph_name, distribution, config.destruction_precision, dim_ratio,
               config.destruction_show_plot, config.destruction_save_plot, config.seed, "DES")

    is_routable(G)
    monitor_gain_knowledge(G, elements_val_id, elements_id_val)