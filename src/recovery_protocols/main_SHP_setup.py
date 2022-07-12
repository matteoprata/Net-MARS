
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
def relaxed_LP_SHP(G, demand_edges, supply_nodes, broken_supply_edges, supply_edges, broken_unk_nodes, broken_unk_edges):
    var_demand_flows = [(i, f) for i, (_, _, f) in enumerate(demand_edges)]   # [(d1, f1), ...]

    # for endpoint source 0, mid 1, destination 2
    # {(n, d): position, }
    var_demand_node_pos = gru.demand_node_position(demand_edges, [name_flow for name_flow, _ in var_demand_flows], G.nodes)

    ###################################################################################################################

    m = Model('netflow')


    # m.setObjective(1, GRB.MAXIMIZE)
    m.params.OutputFlag = 0
    m.params.LogToConsole = 0

    # 1. create: flow variables f_ijh
    flow_var = {}
    for h, dem_val in var_demand_flows:
        for i, j, _ in supply_edges:
            flow_var[h, i, j] = m.addVar(lb=0, ub=G.edges[i, j, co.EdgeType.SUPPLY.value][co.ElemAttr.CAPACITY.value],
                                         vtype=GRB.CONTINUOUS, name='flow_var_{}_{}_{}'.format(h, i, j))

            flow_var[h, j, i] = m.addVar(lb=0, ub=G.edges[i, j, co.EdgeType.SUPPLY.value][co.ElemAttr.CAPACITY.value],
                                         vtype=GRB.CONTINUOUS, name='flow_var_{}_{}_{}'.format(h, j, i))

    # 2. create: perc flow variable a_i
    alpha_var = {}
    for h, _ in var_demand_flows:
        alpha_var[h] = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='alpha_var_{}'.format(h))  # TODO REMEMBER TO SET OK

    # 3. create: repair edge variable y_ij
    rep_edge_var = {}
    for n1, n2, _ in supply_edges:
        var_e = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='rep_edge_var_{}_{}'.format(n1, n2))
        rep_edge_var[n1, n2] = var_e

    # CONSTRAINTS

    epsS = 0
    for i, j, _ in supply_edges:
        n1bro = G.nodes[i][co.ElemAttr.POSTERIOR_BROKEN.value]
        n2bro = G.nodes[j][co.ElemAttr.POSTERIOR_BROKEN.value]
        ebro = G.edges[i, j, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value]

        if n1bro + n2bro + ebro == 0:  # WORKING
            m.addConstr(rep_edge_var[i, j] == 1)

        m.addConstr(quicksum(flow_var[h, i, j] + flow_var[h, j, i] for h, _ in var_demand_flows) <=
                    G.edges[i, j, co.EdgeType.SUPPLY.value][co.ElemAttr.CAPACITY.value] * rep_edge_var[i, j] + epsS * (1 - rep_edge_var[i, j]),
                    'cap_{}_{}'.format(i, j))

    # 2 add: flow conservation constraints
    for h, dem_val in var_demand_flows:  # [(d1, f1)]
        for j in G.nodes:
            to_j, from_j = gru.get_incident_edges_of_node(node=j, edges=supply_edges)  # to_j [(1,600), (50, 600)], [(600, 1), (600, 50)]

            flow_out_j = quicksum(flow_var[h, j, k] for _, k in from_j)  # out flow da j
            flow_in_j = quicksum(flow_var[h, k, j] for k, _ in to_j)     # inner flow da j

            if var_demand_node_pos[j, h] == 0:    # source
                m.addConstr(flow_out_j - flow_in_j == dem_val * alpha_var[h], 'node_%s_%s' % (h, j))

            elif var_demand_node_pos[j, h] == 2:  # destination
                m.addConstr(flow_in_j - flow_out_j == dem_val * alpha_var[h], 'node_%s_%s' % (h, j))

            elif var_demand_node_pos[j, h] == 1:  # intermediate
                m.addConstr(flow_in_j == flow_out_j, 'node_%s_%s' % (h, j))

    # m.addConstr(quicksum(rep_edge_var[n1, n2] for n1, n2, _ in broken_unk_edges) <= R)

    # OBJECTIVE
    epsB = 10**-2
    p0 = quicksum(flow * alpha_var[h] for h, flow in var_demand_flows)
    p1 = quicksum(rep_edge_var[n1, n2] * co.REPAIR_COST * G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value] for n1, n2, _ in broken_unk_edges)
    m.setObjective(p0 - epsB * p1, GRB.MAXIMIZE)

    print("OPTIMIZING...")
    m.update()
    m.optimize()
    print("DONE, RESULT:", GUROBI_STATUS[m.status])
    # debug(m, G, var_demand_flows, broken_unk_edges)
    return var_demand_flows, var_demand_node_pos, supply_edges, m


def debug(m, G, var_demand_flows, broken_unk_edges):
    for h, _ in var_demand_flows:
        hv = m.getVarByName('alpha_var_{}'.format(h)).x
        print(h, hv)

    dems = get_demand_edges(G, is_check_unsatisfied=False, is_residual=False)
    for h, (d1, d2, _) in enumerate(dems):  # enumeration is coherent
        SG = nx.DiGraph()
        for i, j, _ in get_supply_edges(G):
            var_value_v1 = m.getVarByName('flow_var_{}_{}_{}'.format(h, i, j)).x  # edge variable to check if > 0
            var_value_v2 = m.getVarByName('flow_var_{}_{}_{}'.format(h, j, i)).x  # edge variable to check if > 0

            if var_value_v1 > 0:
                SG.add_edge(i, j, capacity=var_value_v1)
                print("flow on", d1, d2, i, j, var_value_v1)

            if var_value_v2 > 0:
                SG.add_edge(j, i, capacity=var_value_v2)
                print("flow on", d1, d2, j, i, var_value_v1)

        poz = nx.spring_layout(SG)
        edge_labels = nx.draw_networkx_edge_labels(SG, pos=poz)
        nx.draw(SG, with_labels=True, pos=poz)
        plt.show()

    for i, j, _ in get_supply_edges(G):
        repa = m.getVarByName('rep_edge_var_{}_{}'.format(i, j)).x
        # if repa > 0:
        print("repair", i, j, repa, (i, j) in broken_unk_edges)

    exit()


def derive_from_optimum_max_shadow_edge(G, demand_flows, var_demand_node_pos, supply_edges, m):
    """ returns the node with maximum in flow """

    total_metric_edge = defaultdict(int)
    # shps, edges = [], []
    if m.status == GRB.status.OPTIMAL:
        for i, j, _ in get_supply_edges(G):
            flow =  sum([m.getVarByName('flow_var_{}_{}_{}'.format(h, i, j)).x for h, _ in demand_flows])
            flow += sum([m.getVarByName('flow_var_{}_{}_{}'.format(h, j, i)).x for h, _ in demand_flows])

            n1bro = G.nodes[i][co.ElemAttr.STATE_TRUTH.value]
            n2bro = G.nodes[j][co.ElemAttr.STATE_TRUTH.value]
            ebro = G.edges[i, j, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value]

            if n1bro + n2bro + ebro > 0 and flow > 0:
                shp = m.getConstrByName('cap_{}_{}'.format(i, j)).Pi
                total_metric_edge[(i, j)] = shp  # + flow
                print("SHP SOLO", shp, 'cap_{}_{}'.format(i, j))

    if len(total_metric_edge) > 0:
        items = total_metric_edge.items()
        items = sorted(items, key=lambda x: x[1], reverse=True)  # [edge, value]
        return items[0][0]
    else:
        return None


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

        # nx.draw(SG)
        # plt.show()

        if d1 in SG.nodes and d2 in SG.nodes and nx.has_path(SG, d1, d2):
            pruned_quant, flow_edge = nx.maximum_flow(SG, d1, d2)
            demand_edges_routed_flow_pp[(d1, d2)] += pruned_quant
            quantity += pruned_quant
            G.edges[d1, d2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value] -= pruned_quant
            for n1 in flow_edge:
                for n2 in flow_edge[n1]:
                    G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value] -= flow_edge[n1][n2]

    return quantity, rep_nodes, rep_edges


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


def run_shp_multi(config):
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
        reset_supply_edges(G)

        print("These are the residual demand edges:")
        print(len(res_demand_edges), res_demand_edges)

        if len(res_demand_edges) == 0:
            demand_log(demands_sat, demand_edges_routed_flow_pp, stats)
            stats_list.append(stats)
            return stats_list

        # SOLVE LP RELAXED
        SG_edges = get_supply_graph(G).edges

        demand_edges = get_demand_edges(G, is_check_unsatisfied=False, is_residual=False)
        broken_supply_nodes = get_element_by_state_KT(G, co.GraphElement.NODE, co.NodeState.BROKEN, co.Knowledge.KNOW)
        unk_supply_nodes = get_element_by_state_KT(G, co.GraphElement.NODE, co.NodeState.UNK, co.Knowledge.KNOW)
        broken_supply_edges = get_element_by_state_KT(G, co.GraphElement.EDGE, co.NodeState.BROKEN, co.Knowledge.KNOW)
        unk_supply_edges = get_element_by_state_KT(G, co.GraphElement.EDGE, co.NodeState.UNK, co.Knowledge.KNOW)
        bro_unk_node = list(set(unk_supply_nodes).union(set(broken_supply_nodes)))
        bro_unk_edge = list(set(unk_supply_edges).union(set(broken_supply_edges)))

        # demand edges // supply edges // nodes // nodes (broken, unk) // supply edges (broken) // supply edges (broken, unk)
        opt = relaxed_LP_SHP(G, demand_edges, G.nodes, broken_supply_edges, SG_edges, bro_unk_node, bro_unk_edge)
        var_demand_flows, var_demand_node_pos, supply_edges, m = opt

        # GET THE EDGE WITH MAX SHP
        edge_to_repair = derive_from_optimum_max_shadow_edge(G, var_demand_flows, var_demand_node_pos, supply_edges, m)

        force_repair = False
        if edge_to_repair is not None:
            rep_nodes, rep_edges = do_repair_full_edge(G, edge_to_repair[0], edge_to_repair[1])
            stats["edge"] += rep_edges
            stats["node"] += rep_nodes
        else:
            force_repair = True

        if len(rep_nodes) > 0:
            # add monitor to v_rep
            monitor_nodes = gu.get_monitor_nodes(G)
            if len(res_demand_edges) > 0 and len(monitor_nodes) < config.monitors_budget:
                moni = rep_nodes[0]
                G.nodes[moni][co.ElemAttr.IS_MONITOR.value] = True
                monitors_stats |= {moni}
                stats["monitors"] |= monitors_stats

            # k-discovery
            K = 2
            SG = get_supply_graph(G)
            reach_k_paths = nx.single_source_shortest_path(SG, rep_nodes[0], cutoff=K)
            for no in reach_k_paths:
                discover_path_truth_limit_broken(G, reach_k_paths[no])
                packet_monitor += 1
                stats["packet_monitoring"] = packet_monitor

        demand_log(demands_sat, demand_edges_routed_flow_pp, stats)
        stats_list.append(stats)
    return stats_list


def run(config):
    return run_shp_multi(config)


def demand_log(demands_sat, demand_edges_routed_flow_pp, stats):
    for ke in demands_sat:
        flow = demand_edges_routed_flow_pp[ke] if ke in demand_edges_routed_flow_pp.keys() else 0
        stats["demands_sat"][ke].append(flow)
