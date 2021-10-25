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
        self.config = config

        self.G = self.init_graph(config.path_to_graph, config.graph_name)
        self.dims_ratio = self.scale_coordinates()
        self.distribution, _, _ = self.destroy()


    @staticmethod
    def init_graph(path_to_graph, graph_name):
        """ Set the labels for the graph components. Inputs graph must maintain the information of the node (ID, longitude x,
        latitude y, state) for the edges (capacity, state).
        """

        def load_graph(path_to_graph, graph_name):
            return nx.MultiGraph(nx.read_gml(os.path.join(path_to_graph, graph_name), label='label'))

        raw_graph = load_graph(path_to_graph, graph_name)
        G = nx.MultiGraph()

        # every element will work by default
        for n1 in raw_graph.nodes:
            G.add_nodes_from([(n1, {co.NodeLabels.STATE.value: co.NodeState.WORKING.name,
                                    co.ElemAttr.LATITUDE.value: float(raw_graph.nodes[n1][co.ElemAttr.LATITUDE.value]),
                                    co.ElemAttr.LONGITUDE.value: float(raw_graph.nodes[n1][co.ElemAttr.LONGITUDE.value])})])

        for n1, n2, gt in raw_graph.edges:
            G.add_edges_from([(n1, n2, {co.NodeLabels.STATE.value: co.NodeState.WORKING.name,
                                        co.ElemAttr.CAPACITY.value: float(raw_graph.edges[n1, n2, gt][co.ElemAttr.CAPACITY.value])
                                        })])

        return G

    def scale_coordinates(self):
        """ Scale graph coordinates to positive [0,1] """

        def get_dimensions():
            """ gets the max/min longitude/latitude and retursn it"""
            coo_lats = [self.G.nodes[n1][co.ElemAttr.LATITUDE.value] for n1 in self.G.nodes]
            coo_long = [self.G.nodes[n1][co.ElemAttr.LONGITUDE.value] for n1 in self.G.nodes]
            return max(coo_long), max(coo_lats), min(coo_long), min(coo_lats)

        max_long, max_lat, min_long, min_lat = get_dimensions()

        assert(max_long > 0 and max_lat > 0)

        max_x_dist = max_long - min(0, min_long)
        max_y_dist = max_lat - min(0, min_lat)
        max_dist, min_dist = max(max_x_dist, max_y_dist), min(max_x_dist, max_y_dist)

        for n1 in self.G.nodes:
            self.G.nodes[n1][co.ElemAttr.LATITUDE.value] = (self.G.nodes[n1][co.ElemAttr.LATITUDE.value] - min(0, min_lat)) / max_dist
            self.G.nodes[n1][co.ElemAttr.LONGITUDE.value] = (self.G.nodes[n1][co.ElemAttr.LONGITUDE.value] - min(0, min_long)) / max_dist

        return {"x": max_x_dist/max_dist, "y": max_y_dist/max_dist}

    def destroy(self):
        """ Handles three type of destruction. """
        if self.config.destruction_type == co.Destruction.GAUSSIAN:
            return dis.gaussian_destruction(self.G, self.config.destruction_precision, self.dims_ratio)
        elif self.config.destruction_type == co.Destruction.UNIFORM:
            return dis.uniform_destruction(self.G)
        elif self.config.destruction_type == co.Destruction.COMPLETE:
            return dis.complete_destruction(self.G)

    def print_graph_info(self):
        print("graph has nodes:", len(self.G.nodes), "and edges:", len(self.G.edges))

    def plot_graph(self):
        gp.plot(self.G, self.distribution, self.config.destruction_precision)

