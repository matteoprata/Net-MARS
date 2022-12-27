import numpy as np

from src.preprocessing.network_routability import *
from src.preprocessing.network_utils import *
import networkx as nx
import sys
np.set_printoptions(threshold=sys.maxsize)

import tqdm
import src.utilities.util_routing_stpath as mxv
import src.preprocessing.network_utils as gu

import scipy.special as sci
import src.utilities.util as util
from itertools import combinations

# OK
# lists of edges are not symmetric, check symmetric edge every time
def broken_paths_without_n(G, paths, broken_paths, broken_edges_T, broken_nodes_T, elements_val_id, elements_id_val, element=None):
    """ Returns the set of paths by ID that contain at least 1 broken element or that do not pass through an element. """

    if broken_paths is None:
        broken_paths = []
        mom_broken, mom_working = [], []
        for path_nodes in paths:  # [[1,2,3,4],
            # adds a path to broken_paths if a node or an edge of the path is broken, i.e. the path is broken
            is_path_broken = False

            # check if path broken, suffices one single element broken
            for i in range(len(path_nodes) - 1):
                n1, n2 = path_nodes[i], path_nodes[i + 1]
                n1, n2 = make_existing_edge(n1, n2)
                is_path_broken = n1 in broken_nodes_T or n2 in broken_nodes_T or (n1, n2, co.EdgeType.SUPPLY.value) in broken_edges_T
                if is_path_broken:
                    mom_broken.append(path_nodes)
                    break

            if not is_path_broken:
                mom_working.append(path_nodes)

        for p in mom_working:
            discover_path(G, p, co.NodeState.WORKING.value)

        for p in mom_broken:
            # NODES
            broken_unk_path_node = [n for n in p if G.nodes[n][co.ElemAttr.POSTERIOR_BROKEN.value] > 0]  # uncertainty or broken

            # EDGES
            broken_unk_path_edge = []
            for i in range(len(p) - 1):
                n1, n2 = p[i], p[i + 1]
                n1, n2 = make_existing_edge(n1, n2)
                if G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value] > 0:
                    broken_unk_path_edge.append((n1, n2))

            if len(broken_unk_path_node) + len(broken_unk_path_edge) == 1:  # single element is broken or unknown
                if len(broken_unk_path_node) == 1:
                    bun = broken_unk_path_node[0]
                    discover_node(G, bun, co.NodeState.BROKEN.value)
                else:
                    bue = broken_unk_path_edge[0]
                    n1, n2 = make_existing_edge(bue[0], bue[1])
                    discover_edge(G, n1, n2, co.NodeState.BROKEN.value)  # bool

            path_els =  [elements_val_id[n] for n in p]
            path_els += [elements_val_id[make_existing_edge(p[i], p[i + 1])] for i in range(len(p) - 1)]
            broken_paths.append(path_els)  # [[n1, e1], []]

    paths_out = broken_paths[:]
    remove_ids = []
    if element is not None:
        # saves the index of the path to remove
        for i in range(len(broken_paths)):
            if elements_val_id[element] in broken_paths[i]:
                remove_ids.append(i)

        paths_out = []
        # keeps the paths that are not traversed by 'element'
        for i in range(len(broken_paths)):
            if i not in remove_ids:
                paths_out.append(broken_paths[i])

    return paths_out


def gain_knowledge_of_n_APPROX(SG, element, element_type, broken_paths, paths, broken_edges_T, broken_nodes_T, elements_val_id, elements_id_val, UNK_prior):
    """ Approximated """

    # failed paths
    bp_without_n = broken_paths_without_n(SG, paths, broken_paths, broken_edges_T, broken_nodes_T, elements_val_id, elements_id_val, element=element)
    bp_with_n = set([tuple(p) for p in broken_paths]) - set([tuple(p) for p in bp_without_n])

    working_edges_P = get_element_by_state_KT(SG, co.GraphElement.EDGE, co.NodeState.WORKING, co.Knowledge.KNOW)
    working_nodes_P = get_element_by_state_KT(SG, co.GraphElement.NODE, co.NodeState.WORKING, co.Knowledge.KNOW)
    working_elements_ids = [elements_val_id[make_existing_edge(n1, n2)] for n1, n2, _ in working_edges_P] + [elements_val_id[n] for n in working_nodes_P]

    if element_type == co.GraphElement.EDGE:
        n1, n2 = make_existing_edge(element[0], element[1])
        ide = elements_val_id[(n1, n2)]
    else:
        ide = elements_val_id[element]

    # if the id of the element is in the working nodes, return that the edge works
    if ide in working_elements_ids:
        return 0

    if len(bp_with_n) == 0:
        return UNK_prior

    # remove working elements in broken paths with n
    new_bp_with_n = []
    for list_elements in bp_with_n:
        bp = []
        #nodes, edges = get_path_elements(path)
        for n in list_elements:
            if n not in working_elements_ids:  # se n è rotto o unk
                bp.append(n)
        new_bp_with_n.append(bp)  # path i cui element func non ci sono

    # 0 if there's at least a path that crosses the element with len 1
    is_broken = sum([1 for p in new_bp_with_n if len(p) == 1]) > 0  # se c'è aleno un path di lunghezza 1 (e quell'uno è per forza l'elemento)  [1,1,1]
    if is_broken:  # is broken
        return 1

    el_bp = set()   # insieme degli elementi rotti o sconosciuti
    for li in new_bp_with_n:
        el_bp |= set(li)

    eps = (len(el_bp) - 1) / len(el_bp)
    P = len(new_bp_with_n) / len(el_bp)

    # print(P, len(new_bp_with_n), len(el_bp))
    # print(min(max(np.floor(P) - 1, 0), 1))
    # print(1-eps / P - P)
    # print()
    # print()
    # P  > 0 // < 1 // 1.5

    if element_type == co.GraphElement.EDGE:
        n1, n2 = element
        prior = SG.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.PRIOR_BROKEN.value]
    else:
        prior = SG.nodes[element][co.ElemAttr.PRIOR_BROKEN.value]

    # T2 = 1 - 1/((P+1)**(prior*10))
    T2 = prior + (1-prior) * (1 - 1/((P+1)**(prior*10)))
    T1 = np.ceil(sum([np.floor(1/len(p)) for p in new_bp_with_n]) / len(new_bp_with_n))
    C = max(T1, T2)

    if not 1 >= C >= 0:
        print("WARNING! Probability was", C)   # TODO FIX!
        exit()
    return C


def heavyside(x):
    return 1 if x >= 0 else 0


def gain_knowledge_of_n_EXACT(SG, element, element_type, broken_paths, broken_paths_padded, tot_els, paths, broken_edges_T,
                              broken_nodes_T, elements_val_id, elements_id_val):
    bp_without_n = broken_paths_without_n(SG, paths, broken_paths, broken_edges_T, broken_nodes_T, elements_val_id, elements_id_val, element=element)

    broken_paths_without_n_padded = np.zeros(shape=(len(bp_without_n), tot_els))
    for i, p in enumerate(bp_without_n):
        broken_paths_without_n_padded[i, :len(p)] = p

    broken_paths_padded = broken_paths_padded.copy()
    working_edges_P = get_element_by_state_KT(SG, co.GraphElement.EDGE, co.NodeState.WORKING, co.Knowledge.KNOW)
    working_nodes_P = get_element_by_state_KT(SG, co.GraphElement.NODE, co.NodeState.WORKING, co.Knowledge.KNOW)

    working_elements_ids = [elements_val_id[make_existing_edge(n1, n2)] for n1, n2, _ in working_edges_P] + [elements_val_id[n] for n in working_nodes_P]

    if element_type == co.GraphElement.EDGE:
        n1, n2 = make_existing_edge(element[0], element[1])
        a_prior = SG.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.PRIOR_BROKEN.value]
    else:
        a_prior = SG.nodes[element][co.ElemAttr.PRIOR_BROKEN.value]

    if len(bp_without_n) == 0:
        num = 1
    else:
        num = probability_broken(broken_paths_without_n_padded, a_prior, working_elements_ids)

    den = probability_broken(broken_paths_padded, a_prior, working_elements_ids)
    prob_n_broken = a_prior * num / den

    error1 = "den gt prior: {}, {}".format(den, a_prior)
    error2 = "probability leak: {}, {}, {}".format(num, den, prob_n_broken)

    # assert a_prior >= den, error1
    assert 1 >= num >= 0 and 1 >= den >= 0 and 1 >= prob_n_broken >= 0, error2
    return prob_n_broken


def dummy_pruning(G):
    """
    pruning for the shortest path
    :param G:
    :param config:
    :return:
    """
    demand_edges_to_repair, demand_edges_routed_flow = [], []
    demand_edges_residual = get_demand_edges(G, is_check_unsatisfied=True, is_capacity=False)
    demand_edges_routed_flow_pp = dict()   # (): 10

    for d_edge in demand_edges_residual:
        d1, d2 = d_edge[0], d_edge[1]
        SG = get_supply_graph(G)
        path = mxv.protocol_stpath(SG, d1, d2)
        if gu.broken_elements_in_path_T(G, path) == 0:
            quantity_pruning = gu.do_prune(G, path)
            demand_edges_routed_flow.append(quantity_pruning)
            demand_edges_routed_flow_pp[(d1, d2)] = quantity_pruning
        else:
            demand_edges_to_repair.append((d1, d2))
    return 0, demand_edges_to_repair, demand_edges_routed_flow, demand_edges_routed_flow_pp


def pruning_monitoring(G, stats_packet_monitoring_so_far, threshold_monitor_message, monitors_map, monitors_connections, monitors_non_connections, last_repaired_demand, config):
    """ PRUNING CORE """
    demand_edges_to_repair = []
    demand_edges_routed_flow = []          # total
    demand_edges_routed_flow_pp = dict()   # (): 10

    # the list of path between demand nodes
    monitoring_paths = []
    stats_packet_monitoring = 0

    # monitors from all the monitor pairs n*(n-1)/2

    monitors = get_monitor_nodes(G)
    demand_nodes = get_demand_nodes(G)
    demand_nodes_residual = get_demand_nodes(G, is_residual=True)
    only_monitors = set(monitors) - demand_nodes

    # n_demand_pairs = int(len(demand_nodes_residual) * (len(demand_nodes_residual) - 1) / 2)
    # n_monitor_couples = len(set_useful_monitors) * (len(set_useful_monitors) - 1) / 2
    # iter_value, flows_to_consider = 0, n_demand_pairs

    handled_pairs, to_handle_pairs = set(), set()
    set_useful_monitors = only_monitors.union(demand_nodes_residual)

    if last_repaired_demand:
        last_m1, last_m2 = last_repaired_demand
        set_useful_monitors -= {last_m1, last_m2}
        # gives priority to edges with endpoints are the last repaired
        set_useful_monitors = [last_m1, last_m2] + list(set_useful_monitors)

    # cretes all possible 2-combinations of useful monitors
    for pair in set(combinations(set_useful_monitors, r=2)):   # pure monitor, and demand nodes (only if exists at least 1 non saturated demand)
        p1, p2 = make_existing_edge(pair[0], pair[1])  # demand edges or monitoring edges
        if is_demand_edge_exists(G, p1, p2):
            res_cap = G.edges[p1, p2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value]
            if res_cap > 0:
                to_handle_pairs.add((p1, p2))  # edges demand not satisfied
        elif config.protocol_monitor_placement != co.ProtocolMonitorPlacement.NONE:
            to_handle_pairs.add((p1, p2))  # edges monitoring

    # PHASE 1
    monitors_connections_merge(monitors_connections, monitors_non_connections, last_repaired_demand)

    halt_monitoring = False
    while len(to_handle_pairs - handled_pairs) > 0:
        SG = get_supply_graph(G)
        priority_paths = {}  # k:path, v:priority
        bubbles = []

        if halt_monitoring:
            break

        print(len(to_handle_pairs-handled_pairs), to_handle_pairs-handled_pairs)

        still_handle = to_handle_pairs - handled_pairs
        # still_handle -= last_m1, last_m2
        # still_handle = list(still_handle)

        for n1_mon, n2_mon in tqdm.tqdm(still_handle, disable=True):
            if stats_packet_monitoring_so_far + stats_packet_monitoring > threshold_monitor_message:  # halt due to monitoring msg
                halt_monitoring = True
                break

            # PRUNING
            to_monitor = True
            if config.protocol_monitor_placement in [co.ProtocolMonitorPlacement.BUDGET_W_REPLACEMENT, co.ProtocolMonitorPlacement.BUDGET]:
                if (config.repairing_mode == co.ProtocolRepairingPath.MIN_COST_BOT_CAP or
                    config.picking_mode == co.ProtocolPickingPath.MIN_COST_BOT_CAP) and not config.is_exhaustive_paths:

                    to_monitor = is_edge_to_monitor(n1_mon, n2_mon, monitors_map,
                                                    monitors_connections, monitors_non_connections, demand_nodes)

            # if no capacitive path exists, abort, this should not happen
            if to_monitor:
                st_path_out = util.safe_exec(mxv.protocol_routing_IP, (SG, n1_mon, n2_mon))  # n1, n2 is not handleable
                stats_packet_monitoring += 1
            else:
                print("Skipped to monitor", n1_mon, n2_mon)
                st_path_out = None, np.inf, None, None

            is_n1_only_monitor = n1_mon not in demand_nodes
            is_n2_only_monitor = n2_mon not in demand_nodes

            if st_path_out is None and not is_n1_only_monitor and not is_n2_only_monitor:
                print("A demand node is ISOLATED! 0 out capacity.", n1_mon, n2_mon)
                return None
            elif st_path_out is None:
                handled_pairs.add((n1_mon, n2_mon))
                continue

            path, metric, rc, is_working = st_path_out
            if path is not None:
                monitoring_paths.append(path)  # <<< probability

            if is_working:  # works AND has capacity
                if is_demand_edge_exists(G, n1_mon, n2_mon):
                    if is_bubble(G, path):   # consider removing on paper
                        bubbles.append(path)
                        print("Urrà, found a bubble!", n1_mon, n2_mon)
                    else:
                        # dem path capacity / residual capacity of the path
                        priority = heuristic_priority_pruning_V2(G, n1_mon, n2_mon, path)
                        priority_paths[tuple(path)] = priority
                    continue

                # monitors are connected PHASE 0
                monitors_connections[n1_mon] |= {n2_mon}
                monitors_connections[n2_mon] |= {n1_mon}

                monitors_non_connections[n1_mon] -= {n2_mon}
                monitors_non_connections[n2_mon] -= {n1_mon}

            else:  # path broken [OR not has capacity] (this second predicate is impossible due to continue above)
                if is_demand_edge_exists(G, n1_mon, n2_mon):
                    demand_edges_to_repair.append((n1_mon, n2_mon))
                    # print("path is broken", n_to_repair_paths)

            # print(monitors_connections)
            handled_pairs.add((n1_mon, n2_mon))

        # --- choose a pruning path ---
        path_to_prune = None
        if len(bubbles) > 0:
            path_to_prune = bubbles[0]  # TODO do prune all the bubbles

        elif len(priority_paths) > 0:
            priority_paths_items = sorted(priority_paths.items(), key=lambda x: x[1], reverse=True)  # path, priority
            path_to_prune = list(priority_paths_items[0][0])  # MAX

        if path_to_prune is not None:
            assert get_path_residual_capacity(G, path_to_prune) > 0
            d1, d2 = make_existing_edge(path_to_prune[0], path_to_prune[-1])
            pruned_quant = do_prune(G, path_to_prune)
            demand_edges_routed_flow.append(pruned_quant)
            demand_edges_routed_flow_pp[(d1, d2)] = pruned_quant

            n1, n2 = path_to_prune[0], path_to_prune[-1]
            if G.edges[n1, n2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value] == 0:
                handled_pairs.add((n1, n2))

    return stats_packet_monitoring, demand_edges_to_repair, demand_edges_routed_flow, monitoring_paths, demand_edges_routed_flow_pp


def pruning_monitoring_dynamic(G, stats_packet_monitoring_so_far, threshold_monitor_message, monitors_map, monitors_connections, monitors_non_connections, last_repaired_demand, config):
    demand_edges_to_repair = []
    demand_edges_routed_flow = []          # total
    demand_edges_routed_flow_pp = dict()   # (): 10
    pruned_paths = []
    # the list of path between demand nodes
    monitoring_paths = []
    stats_packet_monitoring = 0

    # monitors from all the monitor pairs n*(n-1)/2

    monitors = get_monitor_nodes(G)
    demand_nodes = get_demand_nodes(G)
    demand_nodes_residual = get_demand_nodes(G)
    only_monitors = set(monitors) - demand_nodes

    # n_demand_pairs = int(len(demand_nodes_residual) * (len(demand_nodes_residual) - 1) / 2)
    # n_monitor_couples = len(set_useful_monitors) * (len(set_useful_monitors) - 1) / 2
    # iter_value, flows_to_consider = 0, n_demand_pairs

    handled_pairs, to_handle_pairs = set(), set()
    set_useful_monitors = only_monitors.union(demand_nodes_residual)

    if last_repaired_demand:
        last_m1, last_m2 = last_repaired_demand
        set_useful_monitors -= {last_m1, last_m2}
        # gives priority to edges with endpoints taht are the last repaired (gives ORDER)
        set_useful_monitors = [last_m1, last_m2] + list(set_useful_monitors)

    # cretes all possible 2-combinations of useful monitors
    for pair in set(combinations(set_useful_monitors, r=2)):   # pure monitor, and demand nodes (only if exists at least 1 non saturated demand)
        p1, p2 = make_existing_edge(pair[0], pair[1])  # demand edges or monitoring edges
        if is_demand_edge(G, p1, p2) and G.edges[p1, p2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value] == 0:
            continue
        to_handle_pairs.add((p1, p2))

    to_hand = False  # if all demand edges are handled, then ignore handling monitoring paths
    for p1, p2 in to_handle_pairs:
        if is_demand_edge(G, p1, p2):
            to_hand = True
            break

    to_handle_pairs = set() if not to_hand else to_handle_pairs

    # PHASE 1
    monitors_connections_merge(monitors_connections, monitors_non_connections, last_repaired_demand)

    halt_monitoring = False
    while len(to_handle_pairs - handled_pairs) > 0:
        SG = get_supply_graph(G)
        priority_paths = {}  # k:path, v:priority
        bubbles = []

        if halt_monitoring:
            break

        print(len(to_handle_pairs-handled_pairs), to_handle_pairs-handled_pairs)

        still_handle = to_handle_pairs - handled_pairs
        # still_handle -= last_m1, last_m2
        # still_handle = list(still_handle)

        for n1_mon, n2_mon in tqdm.tqdm(still_handle, disable=True):

            st_path_out = util.safe_exec(mxv.protocol_routing_IP, (SG, n1_mon, n2_mon))  # n1, n2 is not handleable
            stats_packet_monitoring += 1

            is_n1_only_monitor = n1_mon not in demand_nodes
            is_n2_only_monitor = n2_mon not in demand_nodes

            if st_path_out is None and not is_n1_only_monitor and not is_n2_only_monitor:
                print("A demand node is ISOLATED! 0 out capacity.", n1_mon, n2_mon)
                return None
            elif st_path_out is None:
                handled_pairs.add((n1_mon, n2_mon))
                continue

            path, metric, rc, is_working = st_path_out
            if path is not None:
                monitoring_paths.append(path)  # <<< probability

            if is_working:  # works AND has capacity
                if is_demand_edge_exists(G, n1_mon, n2_mon):
                    if is_bubble(G, path):   # consider removing on paper
                        bubbles.append(path)
                        print("Urrà, found a bubble!", n1_mon, n2_mon)
                    else:
                        # dem path capacity / residual capacity of the path
                        priority = heuristic_priority_pruning_V2(G, n1_mon, n2_mon, path)
                        priority_paths[tuple(path)] = priority
                    continue

                # monitors are connected PHASE 0
                monitors_connections[n1_mon] |= {n2_mon}
                monitors_connections[n2_mon] |= {n1_mon}

                monitors_non_connections[n1_mon] -= {n2_mon}
                monitors_non_connections[n2_mon] -= {n1_mon}

            else:  # path broken [OR not has capacity] (this second predicate is impossible due to continue above)
                if is_demand_edge_exists(G, n1_mon, n2_mon):
                    demand_edges_to_repair.append((n1_mon, n2_mon))
                    # print("path is broken", n_to_repair_paths)

            # print(monitors_connections)
            handled_pairs.add((n1_mon, n2_mon))

        # --- choose a pruning path ---
        path_to_prune = None
        if len(bubbles) > 0:
            path_to_prune = bubbles[0]  # TODO do prune all the bubbles

        elif len(priority_paths) > 0:
            priority_paths_items = sorted(priority_paths.items(), key=lambda x: x[1], reverse=True)  # path, priority
            path_to_prune = list(priority_paths_items[0][0])  # MAX

        if path_to_prune is not None:
            assert get_path_residual_capacity(G, path_to_prune) > 0
            d1, d2 = make_existing_edge(path_to_prune[0], path_to_prune[-1])
            pruned_quant = do_prune(G, path_to_prune)
            demand_edges_routed_flow.append(pruned_quant)
            demand_edges_routed_flow_pp[(d1, d2)] = pruned_quant

            n1, n2 = path_to_prune[0], path_to_prune[-1]
            if G.edges[n1, n2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value] == 0:
                handled_pairs.add((n1, n2))

            if pruned_quant > 0:
                pruned_paths.append((path_to_prune, pruned_quant))

    return stats_packet_monitoring, demand_edges_to_repair, demand_edges_routed_flow, monitoring_paths, demand_edges_routed_flow_pp, pruned_paths


# def pruning_monitoring_dummy(G, stats_packet_monitoring_so_far, threshold_monitor_message, monitors_map, monitors_connections, monitors_non_connections, last_repaired_demand, config):
#     """ used for all but our algorithm """
#     demand_edges_to_repair = []
#     demand_edges_routed_flow = []          # total
#
#     # the list of path between demand nodes
#     monitoring_paths = []
#     stats_packet_monitoring = 0
#
#     # monitors from all the monitor pairs n*(n-1)/2
#
#     monitors = get_monitor_nodes(G)
#     demand_nodes = get_demand_nodes(G)
#     demand_nodes_residual = get_demand_nodes(G, is_residual=True)
#     only_monitors = set(monitors) - demand_nodes
#
#     handled_pairs, to_handle_pairs = set(), set()
#     set_useful_monitors = only_monitors.union(demand_nodes_residual)
#
#     if last_repaired_demand:
#         last_m1, last_m2 = last_repaired_demand
#         set_useful_monitors -= {last_m1, last_m2}
#         # gives priority to edges with endpoints are the last repaired
#         set_useful_monitors = [last_m1, last_m2] + list(set_useful_monitors)
#
#     # cretes all possible 2-combinations of useful monitors
#     for pair in set(combinations(set_useful_monitors, r=2)):   # pure monitor, and demand nodes (only if exists at least 1 non saturated demand)
#         p1, p2 = make_existing_edge(pair[0], pair[1])  # demand edges or monitoring edges
#         if is_demand_edge_exists(G, p1, p2):
#             res_cap = G.edges[p1, p2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value]
#             if res_cap > 0:
#                 to_handle_pairs.add((p1, p2))  # edges demand not satisfied
#         elif config.protocol_monitor_placement != co.ProtocolMonitorPlacement.NONE:
#             to_handle_pairs.add((p1, p2))  # edges monitoring
#
#     # PHASE 1
#     monitors_connections_merge(monitors_connections, monitors_non_connections, last_repaired_demand)
#
#     halt_monitoring = False
#     while len(to_handle_pairs - handled_pairs) > 0:
#         SG = get_supply_graph(G)
#         priority_paths = {}  # k:path, v:priority
#
#         if halt_monitoring:
#             break
#
#         print(len(to_handle_pairs-handled_pairs), to_handle_pairs-handled_pairs)
#
#         still_handle = to_handle_pairs - handled_pairs
#         # still_handle -= last_m1, last_m2
#         # still_handle = list(still_handle)
#
#         for n1_mon, n2_mon in tqdm.tqdm(still_handle, disable=True):
#             if stats_packet_monitoring_so_far + stats_packet_monitoring > threshold_monitor_message:  # halt due to monitoring msg
#                 halt_monitoring = True
#                 break
#
#             # PRUNING
#             to_monitor = True
#             if not config.is_exhaustive_paths: # Viviana: bisogna mettere questa a true, oppure togliere questo if e il rispettivo to_monitor
#                 to_monitor = is_edge_to_monitor(n1_mon, n2_mon, monitors_map, monitors_connections, monitors_non_connections, demand_nodes)
#
#             # if no capacitive path exists, abort, this should not happen
#             if to_monitor: # Viviana: togli qui
#                 st_path_out = util.safe_exec(mxv.protocol_routing_IP, (SG, n1_mon, n2_mon))  # n1, n2 is not handleable
#                 stats_packet_monitoring += 1
#             else: # Viviana: togli qui
#                 print("Skipped to monitor", n1_mon, n2_mon) # Viviana: togli qui
#                 st_path_out = None, np.inf, None, None # Viviana: togli qui
#
#             is_n1_only_monitor = n1_mon not in demand_nodes
#             is_n2_only_monitor = n2_mon not in demand_nodes
#
#             if st_path_out is None and not is_n1_only_monitor and not is_n2_only_monitor:
#                 print("A demand node is ISOLATED! 0 out capacity.", n1_mon, n2_mon)
#                 return None
#
#             path, metric, rc, is_IP_working = st_path_out
#
#             if path is not None:
#                 monitoring_paths.append(path)  # <<< probability
#
#             if is_IP_working:  # works AND has capacity
#                 # monitors are connected PHASE 0
#                 monitors_connections[n1_mon] |= {n2_mon}
#                 monitors_connections[n2_mon] |= {n1_mon}
#
#                 monitors_non_connections[n1_mon] -= {n2_mon}
#                 monitors_non_connections[n2_mon] -= {n1_mon}
#
#             # print(monitors_connections)
#             handled_pairs.add((n1_mon, n2_mon))
#
#     return stats_packet_monitoring, demand_edges_to_repair, demand_edges_routed_flow, monitoring_paths


def monitors_connections_merge(monitors_connections, monitors_non_connections, last_repaired_demand):
    if last_repaired_demand is not None:
        m1, m2 = last_repaired_demand
        monitors_connections[m1] |= {m2} | monitors_connections[m2]
        monitors_connections[m2] |= {m1} | monitors_connections[m1]

        monitors_non_connections[m1] -= {m2} | monitors_connections[m2]
        monitors_non_connections[m2] -= {m1} | monitors_connections[m1]


def is_edge_to_monitor(n1_mon, n2_mon, monitors_map, monitors_connections, monitors_non_connections, demand_nodes):
    # (MON, MON) (DEM MON) (DEM DEM)

    # ADDED LATER
    is_n1_only_monitor = n1_mon not in demand_nodes
    is_n2_only_monitor = n2_mon not in demand_nodes

    # monitor | monitor > map[m1] \intersect map[m2] non è vuoto
    # monitor | dem     > il monitor deve servire questa domanda (dem, monitor) \in map[model]
    # dem     | monitor > il monitor deve servire questa domanda (dem, monitor) \in map[model]
    # dem     | dem     > tutti casi ok

    func_flat = lambda li: set(list(zip(*li))[0] + list(zip(*li))[1])

    to_handle = False
    if is_n1_only_monitor and is_n2_only_monitor:  # both monitors must serve at least 1 demand edge in common
        if len(set(monitors_map[n1_mon]).intersection(monitors_map[n2_mon])) != 0:
            to_handle = True

    elif is_n1_only_monitor and not is_n2_only_monitor:
        if n2_mon in func_flat(monitors_map[n1_mon]):  # [(a,b)] > [a,b] # n2_mon is demand endpoint, the other is a monitor
            to_handle = True

    elif not is_n1_only_monitor and is_n2_only_monitor:
        if n1_mon in func_flat(monitors_map[n2_mon]):  # n1_mon is demand endpoint
            to_handle = True

    elif not is_n1_only_monitor and not is_n2_only_monitor:  # both demand endpoints
        to_handle = True

    # NEW
    if n1_mon in monitors_non_connections[n2_mon] or n2_mon in monitors_non_connections[n1_mon]:
        # we already know that the two monitors are not connected. There is no need to monitor this demand edge
        to_handle = False

    elif monitors_non_connections[n1_mon].intersection(monitors_connections[n2_mon]) or monitors_non_connections[n2_mon].intersection(monitors_connections[n1_mon]):
        # we know that there is one node connected to n1_mon and not connected to n2_mon, or the viceversa.
        # The couple (n1_mon, n2_mon) must not be monitored as we can infer that they are not connected.
        monitors_connections[n1_mon] -= {n2_mon}.union(monitors_connections[n2_mon])
        monitors_connections[n2_mon] -= {n1_mon}.union(monitors_connections[n1_mon])
        monitors_non_connections[n1_mon] |= {n2_mon}.union(monitors_connections[n2_mon])
        monitors_non_connections[n2_mon] |= {n1_mon}.union(monitors_connections[n1_mon])
        to_handle = False

    return to_handle


def tomography_over_paths(G, elements_val_id, elements_id_val, UNK_prior, monitoring_paths):
    """ TOMOGRAPHY CORE """

    if len(monitoring_paths) < 2:
        print("> No monitoring done. No packets left.")
        return

    SG = get_supply_graph(G)

    broken_edges_T = get_element_by_state_KT(SG, co.GraphElement.EDGE, co.NodeState.BROKEN, co.Knowledge.TRUTH)
    broken_nodes_T = get_element_by_state_KT(SG, co.GraphElement.NODE, co.NodeState.BROKEN, co.Knowledge.TRUTH)

    # n_edges, n_nodes = len(SG.edges), len(SG.nodes)
    # tot_els = n_edges + n_nodes

    # broken_paths  [[n1, n2, ..., e1, e2], []]  path rotti, ma con tutti gli elementi
    bp = broken_paths_without_n(G, monitoring_paths, None, broken_edges_T, broken_nodes_T, elements_val_id, elements_id_val, element=None)
    elements_in_paths = set().union(*bp)   # tutti elementi dei path rotti
    # bp_padded = np.zeros(shape=(len(bp), tot_els), dtype=int)
    # for i, p in enumerate(bp):
    #     bp_padded[i, :len(p)] = p

    # Assign a probability to every element

    new_node_probs = dict()
    new_edge_probs = dict()

    # ASSIGN NODE PROBABILITY BROKEN
    for n1_mon in tqdm.tqdm(G.nodes, disable=True):
        original_posterior = G.nodes[n1_mon][co.ElemAttr.POSTERIOR_BROKEN.value]
        node_id = G.nodes[n1_mon][co.ElemAttr.ID.value]
        if original_posterior not in [0, 1] and node_id in elements_in_paths:
            prob = gain_knowledge_of_n_APPROX(G, n1_mon, co.GraphElement.NODE, bp, monitoring_paths, broken_edges_T,
                                              broken_nodes_T, elements_val_id, elements_id_val, UNK_prior)
            # print((n1), prob)
            new_node_probs[(n1_mon, co.ElemAttr.POSTERIOR_BROKEN.value)] = prob

    # ASSIGN EDGE PROBABILITY BROKEN
    for n1_mon, n2_mon, gt in tqdm.tqdm(G.edges, disable=True):
        if gt == co.EdgeType.SUPPLY.value:
            original_posterior = G.edges[n1_mon, n2_mon, gt][co.ElemAttr.POSTERIOR_BROKEN.value]
            edge_id = G.edges[n1_mon, n2_mon, gt][co.ElemAttr.ID.value]
            if original_posterior not in [0, 1] and edge_id in elements_in_paths:
                prob = gain_knowledge_of_n_APPROX(G, (n1_mon, n2_mon), co.GraphElement.EDGE, bp, monitoring_paths,
                                                  broken_edges_T, broken_nodes_T, elements_val_id, elements_id_val, UNK_prior)
                # print((n1, n2), prob)
                new_edge_probs[(n1_mon, n2_mon, gt, co.ElemAttr.POSTERIOR_BROKEN.value)] = prob

    for k in new_node_probs:
        n1_mon, att = k
        G.nodes[n1_mon][att] = new_node_probs[k]

    for k in new_edge_probs:
        n1_mon, n2_mon, gt, att = k
        G.edges[n1_mon, n2_mon, gt][att] = new_edge_probs[k]


def probability_broken(padded_paths, prior, working_els):
    """ Credits to: Viviana Arrigoni. """

    # TODO: remove rows with same elements
    """ Viviana version! """
    pp = padded_paths.copy()
    if len(working_els) > 0:
        padded_paths[np.isin(padded_paths, working_els)] = 0

    n_paths = padded_paths.shape[0]
    if n_paths == 0:
        return None       # TODO unlikely it executes it

    if n_paths == 1:
        non_zeros_count = np.sum(padded_paths != 0)
        if non_zeros_count == 0:
            return 0
        return 1 - (1 - prior) ** non_zeros_count

    elif n_paths == 2:
        p1, p2 = padded_paths[0, :], padded_paths[1, :]
        p1p2 = np.array(list(set(p1).union(set(p2))))

        l1, l2, l12 = np.sum(p1 > 0), np.sum(p2 > 0), np.sum(p1p2 > 0)
        return 1 - (1 - prior) ** l1 - (1 - prior) ** l2 + (1 - prior) ** l12

    else:
        target = padded_paths.copy()
        target = target[:-1, :]
        target[np.isin(target, padded_paths[-1, :])] = 0
        non_zeros_count_last = np.sum(padded_paths[-1, :] > 0)

        prob = probability_broken(padded_paths[:-1, :], prior, working_els) - \
               probability_broken(target, prior, working_els) * \
               (1 - prior) ** non_zeros_count_last
        return prob


def gain_knowledge_all(G):
    for n in G.nodes:
        G.nodes[n][co.ElemAttr.POSTERIOR_BROKEN.value] = G.nodes[n][co.ElemAttr.STATE_TRUTH.value]

    for n1, n2, et in G.edges:
        G.edges[n1, n2, et][co.ElemAttr.POSTERIOR_BROKEN.value] = G.edges[n1, n2, et][co.ElemAttr.STATE_TRUTH.value]
