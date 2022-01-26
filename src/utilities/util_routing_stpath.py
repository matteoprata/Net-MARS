# Python3 implementation of the approach
import numpy as np

import src.constants as co
import networkx as nx
import src.preprocessing.graph_utils as gu


# Function to print required path
def printpath(src, parent, vertex, target, out):
    # global parent
    if (vertex == src):
        out += [vertex]
        return

    printpath(src, parent, parent[vertex], target, out)
    out += [vertex]
    return out
    # print(vertex, end="\n" if (vertex == target) else "--")


# Function to return the maximum weight
# in the widest path of the given graph
def protocol_routing_stpath(G, src, target, monitors):
    parent = {n: None for n in G.nodes}
    parent[src] = src

    # metric, cap, res_cap, delay, node_id | min su metric, min is in position 0
    container = dict()
    container[src] = (np.inf, np.inf, np.inf, 0)
    for nid in G.nodes:
        if nid != src:
            container[nid] = (np.inf, -np.inf, -np.inf, np.inf)

    while len(container) > 0:
        temp = list(container.items())[0]
        current_src = temp[0]  # node id
        current_src_info = temp[1]  # node id
        del container[current_src]

        if current_src == target:
            break

        # print("Tocca a", current_src, current_src_info)
        # if current_src_info[3] == np.inf:  # isolated node
        #     break

        # print("Passed it. Iter now.", list(G.neighbors(current_src)))
        for neigh in G.neighbors(current_src):
            if neigh == current_src or neigh not in container.keys():
                continue

            # if neigh in set(monitors) - {src, target}:  # TODO RICONSIDERA
            #     continue

            n1, n2 = gu.make_existing_edge(G, current_src, neigh)
            cap = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.CAPACITY.value]

            res_cap = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value]

            is_n1_broken = bool(G.nodes[n1][co.ElemAttr.STATE_TRUTH.value])
            is_n2_broken = bool(G.nodes[n2][co.ElemAttr.STATE_TRUTH.value])
            is_n1n2_broken = bool(G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value])

            if is_n1_broken or is_n2_broken or is_n1n2_broken:
                res_cap = 0

            current_neigh_info = tuple()
            current_neigh_info += container[neigh]

            CA = max(min(current_src_info[1], cap), current_neigh_info[1])
            RC = max(min(current_src_info[2], res_cap), current_neigh_info[2])
            LA = current_src_info[3] + 1
            m = 1/CA + (1/CA) / (RC + co.EPSILON) + LA  # 1 is the hop

            # Relaxation of edge and adding into Priority Queue
            if m < current_neigh_info[0]:
                container[neigh] = (m, CA, RC, LA)
                parent[neigh] = current_src

                container_items = sorted(container.items(), key=lambda x: x[1][0])
                container = {i[0]: i[1] for i in container_items}

                # for i in container:
                #     print(i, container[i])

            if neigh == target:
                for n in container:
                    if m <= container[n][0]:
                        break

    # print(src, target)
    # print(current_src, current_src_info)
    a = printpath(src, parent, target, target, [])
    # print(a)

    return a


def protocol_repair_stpath_OLD(G, src, target):
    parent = {n: None for n in G.nodes}
    node_metric = {n: None for n in G.nodes}
    node_capacity = {n: None for n in G.nodes}

    parent[src] = src

    # metric, cap, res_cap, delay, node_id | min su metric, min is in position 0
    container = dict()
    container[src] = (0, None, np.inf, None)
    for nid in G.nodes:
        if nid != src:
            container[nid] = (np.inf, None, -np.inf, None)

    while len(container) > 0:
        temp = list(container.items())[0]
        current_src = temp[0]  # node id
        current_src_info = temp[1]  # node info
        del container[current_src]

        if current_src == target:
            break

        for neigh in G.neighbors(current_src):
            if neigh == current_src or neigh not in container.keys():
                continue

            n1, n2 = gu.make_existing_edge(G, current_src, neigh)
            res_cap = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value]

            prob_n1 = G.nodes[n1][co.ElemAttr.POSTERIOR_BROKEN.value]
            prob_n2 = G.nodes[n2][co.ElemAttr.POSTERIOR_BROKEN.value]
            prob_n1n2 = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value]

            current_neigh_info = tuple()
            current_neigh_info += container[neigh]

            RC = min(current_src_info[2], res_cap)  # max(min(current_src_info[2], res_cap) , current_neigh_info[2])   # residual

            # m = 1 / RC * (current_src_info[0] * ci + co.REPAIR_COST * ((prob_n1 + prob_n2) / 2 + prob_n1n2))
            cost_prev_but_max_rc = 0 if current_src_info[2] == np.inf else (current_src_info[0] * current_src_info[2])  # cost of before without counting the RC
            cost_now_but_max_rc = co.REPAIR_COST * ((prob_n1 + prob_n2) / 2 + prob_n1n2)  # cost of now without the RC

            m = 1 / (RC + co.EPSILON) * (cost_prev_but_max_rc + cost_now_but_max_rc + 1)

            # Relaxation of edge and adding into Priority Queue
            if m < current_neigh_info[0]:
                # print("ho assegnato a", neigh, "padre", current_src)
                container[neigh] = (m, None, RC, None)
                parent[neigh] = current_src
                node_metric[neigh] = m
                node_capacity[neigh] = RC
                container_items = sorted(container.items(), key=lambda x: x[1][0])  # [(k0, v0), (,)]
                container = {i[0]: i[1] for i in container_items}  # {k0:v0, k1:v1, ...}

            if neigh == target:
                for n in container:
                    if m <= container[n][0]:
                        break

    # print(src, target)
    # print(current_src, current_src_info)
    a = printpath(src, parent, target, target, [])
    metric = node_metric[target]
    capa = node_capacity[target]
    return a, metric, capa


def protocol_repair_stpath(G, src, target):
    parent = {n: None for n in G.nodes}
    node_metric = {n: None for n in G.nodes}
    node_capacity = {n: None for n in G.nodes}

    parent[src] = src

    # metric, cap, res_cap, delay, node_id | min su metric, min is in position 0
    container = dict()
    container[src] = (0, None, np.inf, None)
    for nid in G.nodes:
        if nid != src:
            container[nid] = (np.inf, None, -np.inf, None)

    while len(container) > 0:
        temp = list(container.items())[0]
        current_src = temp[0]  # node id
        current_src_info = temp[1]  # node info
        del container[current_src]

        for neigh in G.neighbors(current_src):
            if neigh not in container.keys():  # avoids loops, no going back, it was already met
                continue

            n1, n2 = gu.make_existing_edge(G, current_src, neigh)
            res_cap = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value]

            prob_n1 = G.nodes[n1][co.ElemAttr.POSTERIOR_BROKEN.value]
            prob_n2 = G.nodes[n2][co.ElemAttr.POSTERIOR_BROKEN.value]
            prob_n1n2 = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value]

            current_neigh_info = tuple()
            current_neigh_info += container[neigh]

            RC = min(current_src_info[2], res_cap)  # max(min(current_src_info[2], res_cap) , current_neigh_info[2])   # residual

            cost_prev_but_max_rc = 0 if current_src_info[2] == np.inf else (current_src_info[0] * current_src_info[2])  # cost of before without counting the RC
            cost_now_but_max_rc = co.REPAIR_COST * ((prob_n1 + prob_n2) / 2 + prob_n1n2) + 1  # cost of now without the RC

            m = 1 / (RC + co.EPSILON) * (cost_prev_but_max_rc + cost_now_but_max_rc)

            # Relaxation of edge and adding into Priority Queue
            if m < current_neigh_info[0]:
                # print("ho assegnato a", neigh, "padre", current_src)
                container[neigh] = (m, None, RC, None)
                parent[neigh] = current_src
                node_metric[neigh] = m
                node_capacity[neigh] = RC

                container_items = sorted(container.items(), key=lambda x: x[1][0])  # [(k0, v0), (,)]
                container = {i[0]: i[1] for i in container_items}  # {k0:v0, k1:v1, ...}

    a = printpath(src, parent, target, target, [])
    metric = node_metric[target]
    capa = node_capacity[target]
    return a, metric, capa



# # Driver code
# if __name__ == '__main__':
#     # Graph representation
#     # graph = [[] for i in range(5)]
#     # no_vertices = 4
#     # # Adding edges to graph
#     #
#     # # Resulting graph
#     # # 1--2
#     # # |  |
#     # # 4--3
#     #
#     # # Note that order in pair is (distance, vertex)
#     # graph[1].append((1, 2))
#     # graph[1].append((2, 4))
#     # graph[2].append((3, 3))
#     # graph[4].append((5, 3))
#
#
#     G = nx.DiGraph()
#     G.add_nodes_from([1, 2, 3, 4])
#     G.add_edges_from([(1, 2), (1, 4), (2, 3), (4, 3)]) #, (2,1), (4,1), (3,2), (3,4)])
#     G.edges[1, 2][co.ElemAttr.CAPACITY.value] = 1
#     G.edges[1, 4][co.ElemAttr.CAPACITY.value] = 2
#     G.edges[4, 3][co.ElemAttr.CAPACITY.value] = 1
#     G.edges[2, 3][co.ElemAttr.CAPACITY.value] = 3
#
#     print(widest_path_problem(G, 1, 3))
#
# # This code is contributed by mohit kumar 29
