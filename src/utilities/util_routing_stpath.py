# Python3 implementation of the approach
import numpy as np

import src.constants as co
import networkx as nx
import src.preprocessing.graph_utils as grau
import src.utilities.util as util


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
def protocol_routing_IP(SG, src, target):

    parent = {n: None for n in SG.nodes}
    parent_residual_cap = {n: None for n in SG.nodes}

    node_metric = {n: None for n in SG.nodes}
    parent[src] = src

    # metric, cap, res_cap, delay, node_id | min su metric, min is in position 0
    container = dict()
    container[src] = (0, np.inf, np.inf, 0, 0)
    for nid in SG.nodes:
        if nid != src:
            container[nid] = (np.inf, -np.inf, -np.inf, np.inf, 0)

    while len(container) > 0:
        temp = list(container.items())[0]
        current_src = temp[0]  # node id
        current_src_info = temp[1]  # node id
        del container[current_src]

        if current_src_info[0] == np.inf:
            break

        # print("Passed it. Iter now.", list(G.neighbors(current_src)))
        for neigh in SG.neighbors(current_src):
            if neigh not in container.keys():
                continue

            n1_st, n2_st = grau.make_existing_edge(SG, current_src, neigh)

            cap = SG.edges[n1_st, n2_st, co.EdgeType.SUPPLY.value][co.ElemAttr.CAPACITY.value]
            res_cap = SG.edges[n1_st, n2_st, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value]

            if res_cap == 0:
                continue

            is_n1_broken = bool(SG.nodes[n1_st][co.ElemAttr.STATE_TRUTH.value])
            is_n2_broken = bool(SG.nodes[n2_st][co.ElemAttr.STATE_TRUTH.value])
            is_n1n2_broken = bool(SG.edges[n1_st, n2_st, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value])

            bro_flag = 1 if is_n1_broken or is_n2_broken or is_n1n2_broken else 0

            current_neigh_info = tuple()
            current_neigh_info += container[neigh]

            BR = max(current_src_info[4], bro_flag)
            CA = min(current_src_info[1], cap)
            RC = min(current_src_info[2], res_cap)
            LA = current_src_info[3] + 1
            m = (1/CA + 1/(CA * RC) + LA) + BR * len(SG.edges)

            # Relaxation of edge and adding into Priority Queue
            if m < current_neigh_info[0]:
                # print("assigned", (src, neigh), m, CA, RC, LA, WO, WO * len(G.edges))
                container[neigh] = (m, CA, RC, LA, BR)
                parent[neigh] = current_src
                parent_residual_cap[neigh] = RC
                node_metric[neigh] = m
                container_items = sorted(container.items(), key=lambda x: x[1][0])
                container = {i[0]: i[1] for i in container_items}

    a = printpath(src, parent, target, target, [])
    metric_out = node_metric[target]
    return a, metric_out, parent_residual_cap[target]


def protocol_repair_AVG_COST(SG, src, target, is_oracle=False):
    parent = {n: None for n in SG.nodes}
    node_metric = {n: None for n in SG.nodes}

    parent[src] = src

    # metric, cap, res_cap, delay, node_id | min su metric, min is in position 0
    container = dict()
    container[src] = (0, None, 0, 0)
    for nid in SG.nodes:
        if nid != src:
            container[nid] = (np.inf, None, -np.inf, None)

    while len(container) > 0:
        temp = list(container.items())[0]
        current_src = temp[0]  # node id
        current_src_info = temp[1]  # node info
        del container[current_src]

        if current_src_info[0] == np.inf:
            break

        for neigh in SG.neighbors(current_src):
            if neigh not in container.keys():  # avoids loops, no going back, it was already met
                continue

            n1, n2 = grau.make_existing_edge(SG, current_src, neigh)
            res_cap = SG.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value]

            if res_cap == 0:
                continue

            attribute = co.ElemAttr.STATE_TRUTH.value if is_oracle else co.ElemAttr.POSTERIOR_BROKEN.value
            prob_n1 = SG.nodes[n1][attribute]
            prob_n2 = SG.nodes[n2][attribute]
            prob_n1n2 = SG.edges[n1, n2, co.EdgeType.SUPPLY.value][attribute]

            current_neigh_info = tuple()
            current_neigh_info += container[neigh]

            AV = (res_cap + current_src_info[3] * current_src_info[2]) / (current_src_info[3] + 1)

            cost_prev_but_max_rc = 0 if current_src_info[2] == np.inf else (current_src_info[0] * current_src_info[2])
            cost_now_but_max_rc = co.REPAIR_COST * (prob_n2 + prob_n1n2)
            m = (cost_prev_but_max_rc + cost_now_but_max_rc) / (AV + co.EPSILON)

            # Relaxation of edge and adding into Priority Queue
            if m < current_neigh_info[0]:
                # print("ho assegnato a", neigh, "padre", current_src)
                container[neigh] = (m, None, AV, current_src_info[3]+1)
                parent[neigh] = current_src
                node_metric[neigh] = m

                container_items = sorted(container.items(), key=lambda x: x[1][0])  # [(k0, v0), (,)]
                container = {i[0]: i[1] for i in container_items}  # {k0:v0, k1:v1, ...}

    a = printpath(src, parent, target, target, [])
    metric = node_metric[target]
    return a, metric, None


def protocol_repair_min_exp_cost(SG, src, target, residual_demand, max_edge_cap, is_oracle):
    parent = {n: None for n in SG.nodes}
    node_metric = {n: None for n in SG.nodes}
    node_capacity = {n: None for n in SG.nodes}

    parent[src] = src

    # metric, cap, res_cap, delay, node_id | min su metric, min is in position 0
    container = dict()

    initial_m_cost = co.REPAIR_COST * SG.nodes[src][co.ElemAttr.POSTERIOR_BROKEN.value]

    container[src] = (initial_m_cost, None, np.inf, 0, initial_m_cost)
    for nid in SG.nodes:
        if nid != src:
            container[nid] = (np.inf, None, -np.inf, -np.inf, -np.inf)

    while len(container) > 0:
        temp = list(container.items())[0]
        current_src = temp[0]  # node id
        current_src_info = temp[1]  # node info
        del container[current_src]

        if current_src_info[0] == np.inf:
            break

        for neigh in SG.neighbors(current_src):
            if neigh not in container.keys():  # avoids loops, no going back, it was already met
                continue

            n1, n2 = grau.make_existing_edge(SG, current_src, neigh)
            res_cap = SG.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value]
            res_cap = util.min_max_normalizer(res_cap, 0, max_edge_cap, 0, 1)

            if res_cap == 0:
                continue  # 0/eps cannot be 0! or loops would happen! if RC == 0, the path cannot be chosen

            attribute = co.ElemAttr.STATE_TRUTH.value if is_oracle else co.ElemAttr.POSTERIOR_BROKEN.value
            prob_n2 = SG.nodes[n2][attribute]
            prob_n1n2 = SG.edges[n1, n2, co.EdgeType.SUPPLY.value][attribute]

            current_neigh_info = tuple()
            current_neigh_info += container[neigh]

            RC = min(min(current_src_info[2], res_cap), residual_demand)

            m = (current_src_info[0] - current_src_info[4]) * (current_src_info[2] if current_src_info[2] != np.inf else 0)
            m = (m + co.REPAIR_COST * prob_n1n2) / RC
            cost_nodes_sofar = current_src_info[4] + co.REPAIR_COST * prob_n2
            m = m + cost_nodes_sofar

            # Relaxation of edge and adding into Priority Queue
            if m < current_neigh_info[0]:
                container[neigh] = (m, None, RC, None, cost_nodes_sofar)
                parent[neigh] = current_src
                node_metric[neigh] = m
                node_capacity[neigh] = RC

                container_items = sorted(container.items(), key=lambda x: x[1][0])  # [(k0, v0), (,)]
                container = {i[0]: i[1] for i in container_items}  # {k0:v0, k1:v1, ...}

    a = printpath(src, parent, target, target, [])
    metric = node_metric[target]
    capa = node_capacity[target]
    return a, metric, capa


def protocol_stpath_capacity(SG, src, target):
    """ Simple shortest path, but avoids 0 capacity paths."""

    parent = {n: None for n in SG.nodes}
    parent[src] = src

    # metric, cap, res_cap, delay, node_id | min su metric, min is in position 0
    container = dict()
    container[src] = (0, np.inf)

    for nid in SG.nodes:
        if nid != src:
            container[nid] = (np.inf, np.inf)

    while len(container) > 0:
        temp = list(container.items())[0]
        current_src = temp[0]  # node id
        current_src_info = temp[1]  # node info
        del container[current_src]

        if current_src_info[0] == np.inf:
            break

        for neigh in SG.neighbors(current_src):
            if neigh not in container.keys():  # avoids loops, no going back, it was already met
                continue

            n1, n2 = grau.make_existing_edge(SG, current_src, neigh)
            res_cap = SG.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value]

            if res_cap == 0:
                continue

            current_neigh_info = tuple()
            current_neigh_info += container[neigh]

            m = current_src_info[0] + 1

            # Relaxation of edge and adding into Priority Queue
            if m < current_neigh_info[0]:
                # print("ho assegnato a", neigh, "padre", current_src)
                container[neigh] = (m, None)
                parent[neigh] = current_src
                container_items = sorted(container.items(), key=lambda x: x[1][0])  # [(k0, v0), (,)]
                container = {i[0]: i[1] for i in container_items}  # {k0:v0, k1:v1, ...}

    a = printpath(src, parent, target, target, [])
    return a


def protocol_repair_cedarlike(SG, src, target):
    parent = {n: None for n in SG.nodes}
    node_metric = {n: None for n in SG.nodes}
    node_capacity = {n: None for n in SG.nodes}

    parent[src] = src

    # metric, cap, res_cap, delay, node_id | min su metric, min is in position 0
    container = dict()
    container[src] = (0, None, np.inf, None)
    for nid in SG.nodes:
        if nid != src:
            container[nid] = (np.inf, None, -np.inf, None)

    while len(container) > 0:
        temp = list(container.items())[0]
        current_src = temp[0]  # node id
        current_src_info = temp[1]  # node info
        del container[current_src]

        if current_src_info[0] == np.inf:
            break

        for neigh in SG.neighbors(current_src):
            if neigh not in container.keys():  # avoids loops, no going back, it was already met
                continue

            n1, n2 = grau.make_existing_edge(SG, current_src, neigh)
            res_cap = SG.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value]

            if res_cap == 0:
                continue

            attribute = co.ElemAttr.POSTERIOR_BROKEN.value
            # CHANGES THIS
            # prob_n1 = 0 if SG.nodes[n1][co.ElemAttr.POSTERIOR_BROKEN.value] == 0 else 1  # sconosciuto o rotto
            prob_n2 = 0 if SG.nodes[n2][attribute] == 0 else 1
            prob_n1n2 = 0 if SG.edges[n1, n2, co.EdgeType.SUPPLY.value][attribute] == 0 else 1

            current_neigh_info = tuple()
            current_neigh_info += container[neigh]

            m = current_src_info[0] + (1-prob_n1n2) / res_cap + co.REPAIR_COST * (prob_n1n2 / res_cap + prob_n2)

            # Relaxation of edge and adding into Priority Queue
            if m < current_neigh_info[0]:
                # print("ho assegnato a", neigh, "padre", current_src)
                container[neigh] = (m, None, None, None)
                parent[neigh] = current_src
                node_metric[neigh] = m

                container_items = sorted(container.items(), key=lambda x: x[1][0])  # [(k0, v0), (,)]
                container = {i[0]: i[1] for i in container_items}  # {k0:v0, k1:v1, ...}

    a = printpath(src, parent, target, target, [])
    metric = node_metric[target]
    capa = node_capacity[target]
    return a, metric, capa