
from src.preprocessing.graph_utils import *
import src.constants as co

# def demand_based_centrality(G, n):
#     """ Computes the centrality for ISP. """
#
#     d_edges = get_demand_edges(G, is_check_unsatisfied=True, is_capacity=True)
#     centrality = 0
#     for d1, d2, c in d_edges:
#         d1, d2 = make_existing_edge(G, d1, d2)
#
#         res_capacity = c
#         Gmom_all = G.copy()  # independent from the other demands
#         all_paths_saturate, capacity_all_paths_saturate = [], []
#         while res_capacity > 0:  # reset residual capacity adding paths
#             path = nx.shortest_path(get_supply_graph(Gmom_all), source=d1, target=d2, weight=co.ElemAttr.WEIGHT.value)  # shortest path
#             path_capacity = get_path_capacity(Gmom_all, path)
#             path_capacity_min = min(path_capacity, res_capacity)
#
#             if path_capacity_min == 0:  # loop otherwise, do not consider 0 capacity edges
#                 set_infinite_weights(Gmom_all, path)
#                 continue
#
#             # print(path, path_capacity, res_capacity, path_capacity_min)
#             demand_pruning(Gmom_all, path, path_capacity_min)
#             res_capacity -= path_capacity_min
#
#             all_paths_saturate.append(path)
#             capacity_all_paths_saturate.append(path_capacity)
#
#         capacity_paths_saturate_through_n = []
#         for i, path in enumerate(all_paths_saturate):
#             path_capacity = capacity_all_paths_saturate[i] if n in path else 0
#             capacity_paths_saturate_through_n.append(path_capacity)
#
#         ratio = sum(capacity_paths_saturate_through_n) / sum(capacity_all_paths_saturate)
#         centrality += ratio * c
#
#     G.nodes[n][co.ElemAttr.CENTRALITY] = centrality





