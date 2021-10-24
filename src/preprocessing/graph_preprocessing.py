import random

import numpy as np

import src.configuration as conf
from src.constants import EdgeType
import networkx as nx
import os

from PIL import Image
from matplotlib import pyplot as plt
import src.plotting.graph_plotting as gp
import src.preprocessing.graph_distruction as dis
import src.constants as co


class TomoCedarNetwork:
    def __init__(self, config: conf.Configuration):
        self.nx_graph = self.__load_graph(config.path_to_graph, config.graph_name)
        self.init_graph()
        self.width = None
        self.height = None
        self.config = config
        self.squared_shape = None

    def init_graph(self):
        """ Set the labels for the graph components"""
        for n1 in self.nx_graph.nodes:
            self.nx_graph.nodes[n1][co.NodeLabels.STATE.value] = co.NodeState.WORKING.name

        for n1, n2, gt_ori in self.nx_graph.edges:
            if gt_ori == co.EdgeType.SUPPLY.value:
                self.nx_graph.edges[n1, n2, gt_ori][co.NodeLabels.STATE.value] = co.NodeState.WORKING.name
            elif gt_ori == co.EdgeType.DEMAND.value:
                self.nx_graph.edges[n1, n2, gt_ori][co.NodeLabels.STATE.value] = co.NodeState.NA.name

    def scale_coordinate(self, zero_one=True):
        max_long, max_lat, min_long, min_lat = self.__get_dimensions()
        abs_max_long, abs_max_lat, abs_min_long, abs_min_lat = abs(max_long), abs(max_lat), abs(min_long), abs(min_lat)

        self.squared_shape = abs_max_long + abs_min_long if max_long > max_lat else abs_max_lat + abs_min_lat

        for node in self.nx_graph.nodes(data=True):
            node[1]["Latitude"] = ((float(node[1]["Latitude"]) + abs_min_lat) / (self.squared_shape if zero_one else 1))
            node[1]["Longitude"] = ((float(node[1]["Longitude"]) + abs_min_long) / (self.squared_shape if zero_one else 1))

    def broke(self):
        self.OUT = dis.gaussian_destruction(self.nx_graph, self.config.destruction_precision)
        #self.OUT = dis.uniform_destruction(self.nx_graph)

    def __get_dimensions(self):
        """ gets the max/min longitude/latitude and retursn it"""
        coo_lats = [float(data['Latitude']) for _, data in self.nx_graph.nodes(data=True)]
        coo_long = [float(data['Longitude']) for _, data in self.nx_graph.nodes(data=True)]

        max_long, min_long = max(coo_long), min(coo_long)
        max_lat, min_lat = max(coo_lats), min(coo_lats)

        return max_long, max_lat, min_long, min_lat

    def print_graph_info(self):
        print("graph has nodes:", len(self.nx_graph.nodes), "and edges:", len(self.nx_graph.edges))

    def plot_graph(self):
        self.scale_coordinate()
        self.broke()
        gp.plot(self.nx_graph, self.OUT, self.config.destruction_precision)

    def __load_graph(self, graph_root, graph_name):
        graph_sup = nx.MultiGraph(nx.read_gml(os.path.join(graph_root, graph_name), label='label'))
        return graph_sup
