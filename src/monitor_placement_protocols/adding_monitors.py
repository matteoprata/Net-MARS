
from src import constants as co
from src.preprocessing import graph_utils as gu
from src.utilities import util_routing_stpath as mxv
import numpy as np


def original_monitoring_add(G, config):
    res_demand_edges = gu.get_demand_edges(G, is_check_unsatisfied=True)
    monitor_nodes = gu.get_monitor_nodes(G)
    monitors = set()
    if len(res_demand_edges) > 0 and len(monitor_nodes) < config.monitors_budget:
        # monitors = monitor_placement_centrality(G, res_demand_edges)
        monitors = __monitor_placement_ours(G, res_demand_edges, config)
    return monitors


def __monitor_placement_ours(G, demand_edges, config):
    # print("Updating centrality")
    bc = None
    bc_centrality = -np.inf

    paths = []
    for n1, n2, _ in demand_edges:
        residual_demand = gu.get_residual_demand(G)
        path, _, _ = mxv.protocol_repair_min_exp_cost(gu.get_supply_graph(G), n1, n2, residual_demand, gu.get_supply_max_capacity(config), config.is_oracle_baseline)
        paths.append(path)

    if len(paths) > 0:
        # between all nodes that are known as working and aren't monitors,
        # the new monitor is the one that resides on most recovery paths
        for n in gu.get_element_by_state_KT(G, co.GraphElement.NODE, co.NodeState.WORKING, co.Knowledge.KNOW):
            if co.ElemAttr.IS_MONITOR.value not in G.nodes[n].keys() or not G.nodes[n][co.ElemAttr.IS_MONITOR.value]:  # is not a monitor already
                node_cent = 0
                for path in paths:
                    node_cent += 1 if n in path else 0

                # print("Node", n, "has centrality", node_cent)
                if node_cent > 0 and node_cent > bc_centrality:
                    bc_centrality = node_cent
                    bc = n

        if bc is not None:
            G.nodes[bc][co.ElemAttr.IS_MONITOR.value] = True
            print("Highest centrality node is", bc)
            return {bc}
        else:
            print("No new added monitor.")
            return set()
    else:
        print("No paths of demand.")
        return set()


def new_monitoring_add(G, config):
    demand_edges = gu.get_demand_edges(G, is_check_unsatisfied=True)
    candidate_monitors = {n: 0 for n in G.nodes}
    candidate_monitors_dem = {n: set() for n in G.nodes}

    paths = []
    for n1, n2, _ in demand_edges:
        residual_demand = gu.get_residual_demand(G)
        path, _, _ = mxv.protocol_repair_min_exp_cost(gu.get_supply_graph(G), n1, n2, residual_demand, gu.get_supply_max_capacity(config), config.is_oracle_baseline)
        paths.append(path)

    if len(paths) > 0 and config.monitors_budget_residual > 0:
        # between all nodes that are known as working and aren't monitors,
        # the new monitor is the one that resides on most recovery paths
        for n in G.nodes:  # gu.get_element_by_state_KT(G, co.GraphElement.NODE, co.NodeState.WORKING, co.Knowledge.KNOW):
            if co.ElemAttr.IS_MONITOR.value not in G.nodes[n].keys() or not G.nodes[n][co.ElemAttr.IS_MONITOR.value]:  # is not a monitor already
                for path in paths:
                    if n in path:
                        candidate_monitors[n] += 1
                        nn1, nn2 = gu.make_existing_edge(G, path[0], path[-1])
                        candidate_monitors_dem[n].add((nn1, nn2))

        candidate_monitors_keys = list(candidate_monitors.keys())[:]

        for k in candidate_monitors_keys:
            if candidate_monitors[k] <= 1:
                del candidate_monitors[k]
                del candidate_monitors_dem[k]

        candidate_monitors_li = sorted(candidate_monitors.items(), key=lambda x: x[1], reverse=True)
        monitors = set([id for id, _ in candidate_monitors_li][0:config.monitors_budget_residual])

        monitors_repaired = []
        for bc in monitors:
            did_repair = gu.do_repair_node(G, bc)
            if did_repair:
                monitors_repaired.append(bc)

            G.nodes[bc][co.ElemAttr.IS_MONITOR.value] = True
            print("Highest centrality node is", bc)

        config.monitors_budget_residual -= len(monitors)
        return monitors, monitors_repaired, candidate_monitors_dem
    else:
        print("There is no monitor to add. Repair paths 0 or monitors budget saturated.")
        return set(), [], dict()


def removing_monitor(G, monitors_map, config):
    """ Removes the monitors that are no more useful. """
    res_demand_edges = gu.get_demand_edges(G, is_check_unsatisfied=True, is_capacity=False)
    monitor_nodes = gu.get_monitor_nodes(G)

    for n in monitor_nodes:
        remove_n = []
        for m_dem in monitors_map[n]:
            remove_n.append(int(m_dem in res_demand_edges))  # if all edges are saturated remove the monitor
        if sum(remove_n) == 0:
            G.nodes[n][co.ElemAttr.IS_MONITOR.value] = False
            del monitors_map[n]
            config.monitors_budget_residual += 1

    return monitors_map


def merge_monitor_maps(monitors_map, moment_monitors_map):
    """ {1:[(1,2)]} {1:[(3,4)], 2:[(5,6)]} > {1:[(1,2), (3,4)], 2:[(5,6)]}"""
    for k in moment_monitors_map:
        if k in monitors_map:
            monitors_map[k] = set(monitors_map[k] | moment_monitors_map[k])
        else:
            monitors_map[k] = moment_monitors_map[k]
    return monitors_map

