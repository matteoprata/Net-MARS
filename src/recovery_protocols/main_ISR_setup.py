
import src.plotting.graph_plotting as pg
from src.preprocessing.graph_preprocessing import *
from src.preprocessing.graph_monitoring import *
from src.preprocessing.graph_utils import *
import src.constants as co

from src.recovery_protocols import finder_recovery_path as frp
from src.recovery_protocols import finder_recovery_path_pick as frpp
from src.monitor_placement_protocols import adding_monitors as mon
import src.utilities.util_routing_stpath as mxv

import time

import src.preprocessing.graph_utils as gru
from gurobipy import *

GUROBI_STATUS = {1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED', 6: 'CUTOFF',
                 7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT', 9: 'TIME_LIMIT', 10: 'SOLUTION_LIMIT', 11: 'INTERRUPTED',
                 12: 'NUMERIC', 13: 'SUBOPTIMAL', 14: 'INPROGRESS', 15: 'USER_OBJ_LIMIT'}


# demand edges // supply edges // nodes // nodes (broken, unk) // supply edges (broken) // supply edges (broken, unk)
def relaxed_LP_multicom(G, demand_edges, supply_nodes, broken_supply_edges, supply_edges, broken_unk_nodes, broken_unk_edges):
    var_demand_flows = [(i, f) for i, (_, _, f) in enumerate(demand_edges)]   # [(d1, f1), ...]

    # for endpoint source 0, mid 1, destination 2
    # {(n, d): position, }
    var_demand_node_pos = gru.demand_node_position(demand_edges, [name_flow for name_flow, _ in var_demand_flows], G.nodes)

    ###################################################################################################################

    m = Model('netflow')

    # m.setObjective(1, GRB.MAXIMIZE)
    m.params.OutputFlag = 0
    m.params.LogToConsole = 0

    # 1. create: flow variables f_ij^h
    flow_var = {}
    for h, dem_val in var_demand_flows:
        for i, j, _ in supply_edges:
            flow_var[h, i, j] = m.addVar(lb=0, ub=G.edges[i, j, co.EdgeType.SUPPLY.value][co.ElemAttr.CAPACITY.value],
                                         vtype=GRB.CONTINUOUS, name='flow_var_{}_{}_{}'.format(h, i, j))

            flow_var[h, j, i] = m.addVar(lb=0, ub=G.edges[i, j, co.EdgeType.SUPPLY.value][co.ElemAttr.CAPACITY.value],
                                         vtype=GRB.CONTINUOUS, name='flow_var_{}_{}_{}'.format(h, j, i))

            # print('flow_var_{}_{}_{}'.format(h, i, j))
            # print('flow_var_{}_{}_{}'.format(h, j, i))

    # 2. create: repair node d_i
    rep_node_var = {}
    for n in supply_nodes:
        rep_node_var[n] = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='rep_node_var_{}'.format(n))

    # 3. create: repair edge d_ij
    rep_edge_var = {}
    for n1, n2, _ in supply_edges:
        var_e = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='rep_edge_var_{},{}'.format(n1, n2))
        rep_edge_var[n1, n2] = var_e

    # 4. create: perc flow
    # alpha_var = {}
    # for h, _ in var_demand_flows:
    #     alpha_var[h] = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='alpha_var_{}'.format(h))

    # print(len(supply_edges - broken_unk_edges), len(supply_edges), len(broken_unk_edges))
    for e in supply_edges:
        i, j, _ = e

        m.addConstr(quicksum(flow_var[h, i, j] + flow_var[h, j, i] for h, _ in var_demand_flows) <=
                    G.edges[i, j, co.EdgeType.SUPPLY.value][co.ElemAttr.CAPACITY.value] * rep_edge_var[i, j],
                    'wor_%s_%s' % (i, j))

    for e0, e1, _ in supply_edges:  # repairing an edge requires reparing the nodes and viceversa
        m.addConstr(rep_edge_var[e0, e1] <= rep_node_var[e1])
        m.addConstr(rep_edge_var[e0, e1] <= rep_node_var[e0])

    # 2 add: flow conservation constraints
    for h, dem_val in var_demand_flows:  # [(d1, f1)]
        for j in G.nodes:
            to_j, from_j = gru.get_incident_edges_of_node(node=j, edges=supply_edges)  # to_j [(1,600), (50, 600)], [(600, 1), (600, 50)]

            flow_out_j = quicksum(flow_var[h, j, k] for _, k in from_j)  # out flow da j
            flow_in_j = quicksum(flow_var[h, k, j] for k, _ in to_j)     # inner flow da j

            if var_demand_node_pos[j, h] == 0:    # source
                m.addConstr(flow_out_j - flow_in_j == dem_val, 'node_%s_%s' % (h, j))
            elif var_demand_node_pos[j, h] == 2:  # destination
                m.addConstr(flow_in_j - flow_out_j == dem_val, 'node_%s_%s' % (h, j))
            elif var_demand_node_pos[j, h] == 1:  # intermediate
                m.addConstr(flow_in_j == flow_out_j, 'node_%s_%s' % (h, j))

    # OBJECTIVE
    m.setObjective(quicksum(rep_node_var[ni] * co.REPAIR_COST * G.nodes[ni][co.ElemAttr.POSTERIOR_BROKEN.value] for ni in broken_unk_nodes) +
                   quicksum(rep_edge_var[n1, n2] * co.REPAIR_COST * G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value] for n1, n2, _ in broken_unk_edges),
                   GRB.MINIMIZE)

    print("OPTIMIZING...")
    m.update()
    m.optimize()
    print("DONE, RESULT:", GUROBI_STATUS[m.status])
    return var_demand_flows, var_demand_node_pos, supply_edges, m


def derive_from_optimum_max_flow_node(G, demand_flows, var_demand_node_pos, supply_edges, m):
    """ returns the node with maximum in flow """
    total_flow_node = defaultdict(int)  # all the nodes associated the flow
    path_nodes, path_edges = set(), set()

    if m.status == GRB.status.OPTIMAL:
        for h, _ in enumerate(demand_flows):
            for i, j, _ in supply_edges:

                var_value_v1 = m.getVarByName('flow_var_{}_{}_{}'.format(h, i, j))  # edge variable to check if > 0
                var_value_v2 = m.getVarByName('flow_var_{}_{}_{}'.format(h, j, i))  # edge variable to check if > 0

                if var_value_v1.x > 0 or var_value_v2.x > 0:
                    # print("positive var", h, i, j, var_value_v1.x, var_value_v2.x, "pbro:", G.nodes[i][co.ElemAttr.POSTERIOR_BROKEN.value])

                    # TODO: LIL strong also in SHP
                    n1bro = G.nodes[i][co.ElemAttr.STATE_TRUTH.value]
                    n2bro = G.nodes[j][co.ElemAttr.STATE_TRUTH.value]
                    ebro = G.edges[i, j, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value]

                    if n1bro + n2bro + ebro > 0:  # broken or unk

                        if var_demand_node_pos[i, h] == 0:
                            total_flow_node[i] += var_value_v1.x
                        else:  # in any other case
                            total_flow_node[i] += var_value_v2.x

                        if var_demand_node_pos[j, h] == 0:
                            total_flow_node[j] += var_value_v2.x
                        else:  # in any other case
                            total_flow_node[j] += var_value_v1.x

                        path_nodes.add(i)
                        path_nodes.add(j)
                        path_edges.add((i, j))

    # choose the node that has max flow
    max_flow_node, max_flow = None, -np.inf
    for k in total_flow_node:    # total_flow_node is empy when path nodes è vuoto o tutto funziona
        if total_flow_node[k] > max_flow:
            max_flow = total_flow_node[k]
            max_flow_node = k

    # print(max_flow_node, max_flow, path_nodes, total_flow_node)
    if max_flow_node is None:
        return None, path_nodes, path_edges

    # broken_flow_nodes = total_flow_node.keys()
    # print("broken to choose", max_flow_node, max_flow, broken_flow_nodes)
    return max_flow_node, path_nodes, path_edges


def derive_from_optimum_repairs(G, m):
    path_nodes, path_edges = set(), set()
    for v in m.getVars():
        if v.X >= .5:  # means repair_node, repair_edge
            if str(v.VarName).startswith("rep_node_var_"):
                node = int(str(v.VarName).split("_")[-1])
                # print(node, G.nodes[node][co.ElemAttr.POSTERIOR_BROKEN.value])
                if G.nodes[node][co.ElemAttr.POSTERIOR_BROKEN.value] > 0:
                    path_nodes.add(node)

            elif str(v.VarName).startswith("rep_edge_var_"):
                edge = str(v.VarName).split("_")[-1].split(",")
                edd = (int(edge[0]), int(edge[1]))
                # print(edd, G.edges[edd[0], edd[1], co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value])
                if G.edges[edd[0], edd[1], co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value] > 0:
                    path_edges.add(edd)

    path_nodes |= set([e1 for e1, _ in path_edges]) | set([e2 for e2, _ in path_edges])
    return path_nodes, path_edges


def isr_srt(G):
    """ ISR policy assumes that weights on the graph are updated.
    Shortest path with weight the "minimum expected cost" only for demand paths containing broken unk elements. """
    nodes, edges, paths = set(), set(), list()
    SG = get_supply_graph(G)
    for ed in get_demand_edges(G, is_check_unsatisfied=True):
        path = nx.shortest_path(SG, ed[0], ed[1], weight=co.ElemAttr.WEIGHT.value)
        is_unk_bro_path = False
        for nd in path:
            if G.nodes[nd][co.ElemAttr.POSTERIOR_BROKEN.value] > 0:  # nodes broken or unk
                nodes.add(nd)
                is_unk_bro_path = True

        for i in range(len(path)-1):
            n1, n2 = path[i], path[i+1]
            n1, n2 = make_existing_edge(G, n1, n2)
            if G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value] > 0:  # edges broken or unk
                edges |= {(n1, n2)}
                is_unk_bro_path = True

        if is_unk_bro_path:
            paths.append(path)
    return nodes, edges, paths


def update_graph_probabilities(G):
    """ shortest path weights, probabilistic """
    for n1, n2, et in G.edges:
        if et == co.EdgeType.SUPPLY.value:
            pn1 = G.nodes[n1][co.ElemAttr.POSTERIOR_BROKEN.value]
            pn2 = G.nodes[n2][co.ElemAttr.POSTERIOR_BROKEN.value]
            pn1n2 = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value]
            cost = co.REPAIR_COST * ((pn1 + pn2) / 2 + pn1n2)
            G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.WEIGHT.value] = cost


def shortest_through_v(v, paths_tot):
    paths, endpoints = [], []
    for pa in paths_tot:
        if v in pa:
            paths.append(v)
            endpoints.append((pa[0], pa[-1]))
    return paths, endpoints


def flow_var_pruning_demand(G, m, force_repair, demand_edges_routed_flow_pp):
    # set the infinite weight for the 0 capacity edges

    quantity = 0
    rep_nodes, rep_edges = [], []

    if m is None:
        return quantity, rep_nodes, rep_edges

    SGOut = get_supply_graph(G)
    for h, (d1, d2, _) in enumerate(get_demand_edges(G, is_check_unsatisfied=False, is_residual=False)):  # enumeration is coherent
        SG = nx.DiGraph()
        for i, j, _ in SGOut.edges:
            var_value_v1 = m.getVarByName('flow_var_{}_{}_{}'.format(h, i, j)).x  # edge variable to check if > 0
            var_value_v2 = m.getVarByName('flow_var_{}_{}_{}'.format(h, j, i)).x  # edge variable to check if > 0

            n1bro = G.nodes[i][co.ElemAttr.STATE_TRUTH.value]
            n2bro = G.nodes[j][co.ElemAttr.STATE_TRUTH.value]
            ebro = G.edges[i, j, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value]

            if var_value_v1 > 0 or var_value_v2 > 0:
                if n1bro + n2bro + ebro == 0 or force_repair:
                    # print("added", h, d1, d2, i, j, var_value_v1, var_value_v2, n1bro, n2bro, ebro)
                    if force_repair:  # this because sometimes every node is ok, but some edges of working nodes are not repaired
                        repn, repe = do_repair_full_edge(G, i, j)
                        rep_nodes += repn
                        rep_edges += repe

                    if var_value_v2 > 0:
                        SG.add_edge(j, i, capacity=var_value_v2)

                    if var_value_v1 > 0:
                        SG.add_edge(i, j, capacity=var_value_v1)

        if d1 in SG.nodes and d2 in SG.nodes and nx.has_path(SG, d1, d2):
            pruned_quant, flow_edge = nx.maximum_flow(SG, d1, d2)
            demand_edges_routed_flow_pp[(d1, d2)] += pruned_quant
            quantity += pruned_quant
            G.edges[d1, d2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value] -= pruned_quant
            for n1 in flow_edge:
                for n2 in flow_edge[n1]:
                    G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value] -= flow_edge[n1][n2]
    return quantity, rep_nodes, rep_edges


def isr_pruning_demand(G, demand_edges_routed_flow_pp):
    # set the infinite weight for the 0 capacity edges
    SG = get_supply_graph(G)
    quantity = 0
    for n1, n2, _ in SG.edges:
        cap = SG.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value]
        if cap <= 0:
            SG.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.WEIGHT.value] = np.inf

    # prune on least cost paths
    for d1, d2, _ in get_demand_edges(G, is_check_unsatisfied=True):
        path = nx.shortest_path(SG, d1, d2, weight=co.ElemAttr.WEIGHT.value)
        if is_path_working(G, path):
            pruned_quant = do_prune(G, path)
            quantity += pruned_quant
            demand_edges_routed_flow_pp[(d1, d2)] += pruned_quant
            discover_path(G, path, co.NodeState.WORKING.value)
    return quantity


def remaining_demand_endpoints(G, d_edges):
    cap = 0
    for e1, e2 in d_edges:
        e1, e2 = make_existing_edge(G, e1, e2)
        cap += G.edges[e1, e2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value]
    return cap


def run_header(config):
    stats_list = []

    # read graph and print stats
    G, elements_val_id, elements_id_val = init_graph(co.PATH_TO_GRAPH, config.graph_path, config.supply_capacity,
                                                     config)
    print_graph_info(G)

    # normalize coordinates and break components
    dim_ratio = scale_coordinates(G)

    distribution, broken_nodes, broken_edges, perc_broken_elements = destroy(G, config.destruction_type,
                                                                             config.destruction_precision, dim_ratio,
                                                                             config.destruction_width,
                                                                             config.n_destruction, config.graph_dataset,
                                                                             config.seed,
                                                                             ratio=config.destruction_quantity,
                                                                             config=config)

    # add_demand_endpoints
    if config.is_demand_clique:
        add_demand_clique(G, config.n_demand_clique, config.demand_capacity, config)
    else:
        add_demand_pairs(G, config.n_demand_pairs, config.demand_capacity, config)

    # hypothetical routability
    if not is_feasible(G, is_fake_fixed=True):
        print(
            "This instance is not solvable. Check the number of demand edges, theirs and supply links capacity.\n\n\n")
        return

    # repair demand edges
    demand_node = get_demand_nodes(G)
    for dn in demand_node:
        do_repair_node(G, dn)
        # INITIAL NODES repairs are not counted in the stats

    iter = 0
    # true ruotability

    routed_flow = 0
    packet_monitor = 0
    monitors_stats = set()
    demands_sat = {d: [] for d in
                   get_demand_edges(G, is_capacity=False)}  # d1: [0, 1, 1, 0, 10] // demands_sat[d].append(0)

    # if config.monitoring_type == co.PriorKnowledge.FULL:
    #     gain_knowledge_all(G)

    assert config.monitors_budget == -1 or config.monitors_budget >= len(get_demand_nodes(G)), \
        "budget is {}, demand nodes are {}".format(config.monitors_budget, len(get_demand_nodes(G)))

    if config.monitors_budget == -1:  # -1 budget means to set automatically as get_demand_nodes(G)
        config.monitors_budget = get_demand_nodes(G)

    return G, stats_list, monitors_stats, packet_monitor, demands_sat, routed_flow, iter


def run_isr_st(config):
    G, stats_list, monitors_stats, packet_monitor, demands_sat, routed_flow, iter = run_header(config)

    # start of the protocol
    while len(get_demand_edges(G, is_check_unsatisfied=True)) > 0:
        # go on if there are demand edges to satisfy, and still is_feasible
        demand_edges_routed_flow_pp = defaultdict(int)  # (d_edge): flow

        print("\n\n", "#" * 40, "BEGIN ITERATION", "#" * 40)

        # check if the graph is still routbale on tot graph,
        if not is_feasible(G, is_fake_fixed=True):
            print("This instance is no more routable!")
            return stats_list

        iter += 1
        print("ITER", iter)

        # packet_monitor -- monitors paced up to iteration i
        # monitors -- monitors placed up to now (no duplicates)
        stats = {"iter": iter,
                 "node": [],
                 "edge": [],
                 "flow": 0,
                 "monitors": monitors_stats,
                 "packet_monitoring": packet_monitor,
                 "demands_sat": demands_sat}

        update_graph_probabilities(G)
        quantity = isr_pruning_demand(G, demand_edges_routed_flow_pp)
        routed_flow += quantity
        stats["flow"] = routed_flow

        paths_nodes, paths_edges, paths_tot = isr_srt(G)
        paths_elements = paths_nodes.union(paths_edges)
        paths_nodes, paths_edges = list(paths_nodes), list(paths_edges)  # nodes for which the state is broken or unknown

        print(paths_nodes)
        print(paths_edges)

        if len(paths_elements) == 0:  # it may be all working elements, do not return!, rather go on
            print("Process completed!", get_residual_demand(G), get_demand_edges(G, True, True, True))
            demand_log(demands_sat, demand_edges_routed_flow_pp, stats)
            stats_list.append(stats)
            return stats_list
        else:
            if len(paths_nodes) == 0 and len(paths_edges) > 0:
                rep_nodes, rep_edges = [], []
                for ed1, ed2 in paths_edges:
                    rep_b = do_repair_edge(G, ed1, ed2)
                    rep_edges.append(rep_b)

                rep_edges = [rp for rp in rep_edges if rp is not None]
                stats["edge"] += rep_edges

                demand_log(demands_sat, demand_edges_routed_flow_pp, stats)
                stats_list.append(stats)
                continue

            # find the best node to repair based on max flow
            r_rem_tot = get_residual_demand(G)
            values_of_v = []
            for v in paths_nodes:
                _, endpoints = shortest_through_v(v, paths_tot)  # per MULTI commodity usare il nodo i che max sum_j fij + f(ji)
                f_rem_v = remaining_demand_endpoints(G, endpoints)
                nv = f_rem_v / r_rem_tot
                values_of_v.append(nv)

            node_rep_id = np.argmax(values_of_v)
            node_rep = paths_nodes[node_rep_id]
            # todo: break ties

            rep_a = do_repair_node(G, node_rep)
            if rep_a is not None:
                stats["edge"] += [rep_a]

            for ed1, ed2 in paths_edges:
                if node_rep in (ed1, ed2):
                    rep_nodes, rep_edges = do_repair_full_edge(G, ed1, ed2)
                    stats["edge"] += rep_edges
                    stats["node"] += rep_nodes

            # print residual edges
            res_demand_edges = gu.get_demand_edges(G, is_check_unsatisfied=True)
            monitor_nodes = gu.get_monitor_nodes(G)
            print("These are the residual demand edges:")
            print(len(res_demand_edges), res_demand_edges)

            # add monitor to v_rep
            if len(res_demand_edges) > 0 and len(monitor_nodes) < config.monitors_budget:
                monitors_stats |= {node_rep}
                stats["monitors"] |= monitors_stats

            # k-discovery
            K = 2
            SG = get_supply_graph(G)
            reach_k_paths = nx.single_source_shortest_path(SG, node_rep, cutoff=K)
            for no in reach_k_paths:
                discover_path_truth_limit_broken(G, reach_k_paths[no])
                packet_monitor += 1
                stats["packet_monitoring"] = packet_monitor

        demand_log(demands_sat, demand_edges_routed_flow_pp, stats)
        stats_list.append(stats)

    return stats_list


def run_isr_multi(config):
    G, stats_list, monitors_stats, packet_monitor, demands_sat, routed_flow, iter = run_header(config)
    m, force_repair = None, False

    # start of the protocol
    while len(get_demand_edges(G, is_check_unsatisfied=True)) > 0:
        # go on if there are demand edges to satisfy, and still is_feasible
        demand_edges_routed_flow_pp = defaultdict(int)  # (d_edge): flow

        print("\n\n", "#" * 40, "BEGIN ITERATION", "#" * 40)

        # check if the graph is still routbale on tot graph,
        if not is_feasible(G, is_fake_fixed=True):
            print("This instance is no more routable!")
            return stats_list

        iter += 1
        print("ITER", iter)

        # packet_monitor -- monitors paced up to iteration i
        # monitors -- monitors placed up to now (no duplicates)
        stats = {"iter": iter,
                 "node": [],
                 "edge": [],
                 "flow": 0,
                 "monitors": monitors_stats,
                 "packet_monitoring": packet_monitor,
                 "demands_sat": demands_sat}

        # PRUNING
        quantity, rep_nodes, rep_edges = flow_var_pruning_demand(G, m, force_repair, demand_edges_routed_flow_pp)
        stats["edge"] += rep_edges
        stats["node"] += rep_nodes
        routed_flow += quantity
        stats["flow"] = routed_flow

        res_demand_edges = gu.get_demand_edges(G, is_check_unsatisfied=True)
        monitor_nodes = gu.get_monitor_nodes(G)

        reset_supply_edges(G)

        print("These are the residual demand edges:")
        print(len(res_demand_edges), res_demand_edges)

        if len(res_demand_edges) == 0:
            demand_log(demands_sat, demand_edges_routed_flow_pp, stats)
            stats_list.append(stats)
            return stats_list

        # OPTIM
        SG_edges = get_supply_graph(G).edges

        demand_edges = get_demand_edges(G, is_check_unsatisfied=False, is_residual=False)
        broken_supply_nodes = get_element_by_state_KT(G, co.GraphElement.NODE, co.NodeState.BROKEN, co.Knowledge.KNOW)
        unk_supply_nodes = get_element_by_state_KT(G, co.GraphElement.NODE, co.NodeState.UNK, co.Knowledge.KNOW)
        broken_supply_edges = get_element_by_state_KT(G, co.GraphElement.EDGE, co.NodeState.BROKEN, co.Knowledge.KNOW)
        unk_supply_edges = get_element_by_state_KT(G, co.GraphElement.EDGE, co.NodeState.UNK, co.Knowledge.KNOW)
        bro_unk_node = list(set(unk_supply_nodes).union(set(broken_supply_nodes)))
        bro_unk_edge = list(set(unk_supply_edges).union(set(broken_supply_edges)))

        # demand edges // supply edges // nodes // nodes (broken, unk) // supply edges (broken) // supply edges (broken, unk)
        opt = relaxed_LP_multicom(G, demand_edges, G.nodes, broken_supply_edges, SG_edges, bro_unk_node, bro_unk_edge)
        var_demand_flows, var_demand_node_pos, supply_edges, m = opt

        # vv = get_element_by_state_KT(G, co.GraphElement.NODE, co.NodeState.BROKEN, co.Knowledge.TRUTH)
        # print("broken", len(vv), vv)

        node_rep, path_nodes, path_edges = derive_from_optimum_max_flow_node(G, var_demand_flows, var_demand_node_pos, supply_edges, m)
        # todo break ties

        print(node_rep, path_nodes, path_edges)

        # paths_elements = set(path_nodes).union(set(path_edges))

        force_repair = False
        if node_rep is None:
            force_repair = True
        else:
            # repair
            rep_a = do_repair_node(G, node_rep)
            if rep_a is not None:
                stats["edge"] += [rep_a]

            for ed1, ed2 in path_edges:
                if node_rep in (ed1, ed2):
                    rep_nodes, rep_edges = do_repair_full_edge(G, ed1, ed2)
                    stats["edge"] += rep_edges
                    stats["node"] += rep_nodes

        if node_rep is not None:
            # add monitor to v_rep
            if len(res_demand_edges) > 0 and len(monitor_nodes) < config.monitors_budget:
                monitors_stats |= {node_rep}
                stats["monitors"] |= monitors_stats

            # k-discovery
            K = 2
            SG = get_supply_graph(G)
            reach_k_paths = nx.single_source_shortest_path(SG, node_rep, cutoff=K)
            for no in reach_k_paths:
                discover_path_truth_limit_broken(G, reach_k_paths[no])
                packet_monitor += 1
                stats["packet_monitoring"] = packet_monitor

        demand_log(demands_sat, demand_edges_routed_flow_pp, stats)
        stats_list.append(stats)
    return stats_list


def run(config):

    if config.algo_name == co.AlgoName.ISR:
        return run_isr_st(config)
    elif config.algo_name == co.AlgoName.ISR_MULTICOM:
        return run_isr_multi(config)
    else:
        exit()


def demand_log(demands_sat, demand_edges_routed_flow_pp, stats):
    for ke in demands_sat:
        flow = demand_edges_routed_flow_pp[ke] if ke in demand_edges_routed_flow_pp.keys() else 0
        stats["demands_sat"][ke].append(flow)
