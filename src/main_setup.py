
import networkx as nx

import src.preprocessing.graph_preprocessing as gp
from src.preprocessing.graph_preprocessing import TomoCedarNetwork


def run(config):
    tm = TomoCedarNetwork(config)
    tm.print_graph_info()
    tm.plot_graph()

    # graph_sup = gp.load_graph(config.path_to_graph, config.graph_name)  # load supply graph
    # gp.print_graph_info(graph_sup)
    #
    # nodes_destroyed, edges_destroyed = gp.destroy_graph(G) if not "random" in config.graph_name else destroy_ALL_graph(G)
    #
    # # ---- start - stats sulle distruzioni
    # total_elements = G.number_of_nodes() + G.number_of_edges()
    # broken_elements = len(nodes_destroyed) + len(edges_destroyed)
    #
    # percentage_disruption = broken_elements / total_elements * 100
    #
    # print("Percentage of broke elements:", percentage_disruption,
    #       "broken arcs:", str(len(edges_destroyed)) + "/" + str(G.number_of_edges()),
    #       "broken nodes:", str(len(nodes_destroyed)) + "/" + str(G.number_of_nodes()),
    #       "broken elements:", str(broken_elements) + "/" + str(total_elements))

