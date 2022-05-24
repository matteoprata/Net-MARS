
import src.plotting.graph_plotting as pg
from src.preprocessing.graph_preprocessing import *
from src.preprocessing.graph_monitoring import *
from src.preprocessing.graph_utils import *
import src.constants as co

from src.recovery_protocols import finder_recovery_path as frp
from src.recovery_protocols import finder_recovery_path_pick as frpp
from src.monitor_placement_protocols import adding_monitors as mon

import time

import src.preprocessing.graph_utils as gru
from gurobipy import *


GUROBI_STATUS = {1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED', 6: 'CUTOFF',
                 7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT', 9: 'TIME_LIMIT', 10: 'SOLUTION_LIMIT', 11: 'INTERRUPTED',
                 12: 'NUMERIC', 13: 'SUBOPTIMAL', 14: 'INPROGRESS', 15: 'USER_OBJ_LIMIT'}


def shp_optimum(G, demand_edges, supply_nodes, broken_supply_edges, supply_edges, broken_unk_nodes, broken_unk_edges):

    # print(supply_edges)
    # print(broken_supply_edges)
    # print(broken_unk_nodes)
    # print(broken_unk_edges)
    # print()
    # exit()

    var_demand_flows = []
    for i, (n1, n2, f) in enumerate(demand_edges):
        var_demand_flows.append((i, f))

    # for endpoint source 0, mid 1, destination 2
    var_demand_node_pos = gru.demand_node_position(demand_edges, [name_flow for name_flow, _ in var_demand_flows], G.nodes)

    ###################################################################################################################

    m = Model('netflow')

    # m.setObjective(1, GRB.MAXIMIZE)
    m.params.OutputFlag = 0
    m.params.LogToConsole = 0
    constraints_names = []

    # 1. create: flow variables f_ij^h
    flow_var = {}
    for h, dem_val in var_demand_flows:
        for i, j, _ in supply_edges:
            flow_var[h, i, j] = m.addVar(lb=0, ub=min(G.edges[i, j, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value], dem_val),
                                         vtype=GRB.CONTINUOUS, name='flow_var_{}_{}_{}'.format(h, i, j))

            flow_var[h, j, i] = m.addVar(lb=0, ub=min(G.edges[i, j, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value], dem_val) ,
                                         vtype=GRB.CONTINUOUS, name='flow_var_{}_{}_{}'.format(h, i, j))

    flow_quantity = {}
    for i, (n1, n2, f) in enumerate(demand_edges):
        flow_quantity[i] = m.addVar(lb=0, ub=G.edges[n1, n2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value],
                                     vtype=GRB.CONTINUOUS, name='dem_flow_var_{}'.format(i))

    # 3. create: repair edge d_ij
    rep_edge_var = {}
    for n1, n2, _ in supply_edges:
        var_e = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='rep_edge_var_{},{}'.format(n1, n2))
        rep_edge_var[n1, n2] = var_e

    # CONSTRAINTS
    # 1. add: edge capacity constraints
    use_constraints = 0
    for sed in supply_edges:
        i, j, _ = sed
        for h, dem_val in var_demand_flows:
            m.addConstr(flow_var[h, i, j] <= (rep_edge_var[i, j] * min(
                G.edges[i, j, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value], dem_val)),
                        "topi_{}_{}".format(i, j))

            m.addConstr(flow_var[h, j, i] <= (rep_edge_var[i, j] * min(
                G.edges[i, j, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value], dem_val)),
                        "topi_{}_{}".format(j, i))

            constraints_names.append((use_constraints, (i, j)))
            use_constraints += 1
            constraints_names.append((use_constraints, (j, i)))
            use_constraints += 1

    for sed in supply_edges:
        i, j, _ = sed

        if sed not in broken_supply_edges:
            m.addConstr(rep_edge_var[i, j] == 0)
            constraints_names.append(None)

    # budget
    m.addConstr(quicksum(rep_edge_var[i, j] for i, j, _ in broken_supply_edges) <= 20)
    constraints_names.append(None)

    # 2 add: flow conservation constraints
    for h, dem_val in var_demand_flows:
        for j in G.nodes:
            to_j, from_j = gru.get_incident_edges_of_node(node=j, edges=supply_edges)

            flow_out_j = quicksum(flow_var[h, j, k] for _, k in from_j)  # out flow da j
            flow_in_j = quicksum(flow_var[h, k, j] for k, _ in to_j)     # inner flow da j

            if var_demand_node_pos[j, h] == 0:    # source
                m.addConstr(flow_out_j - flow_in_j == flow_quantity[h], 'node_%s_%s' % (h, j))
                constraints_names.append(None)
            elif var_demand_node_pos[j, h] == 2:  # destination
                m.addConstr(flow_in_j - flow_out_j == flow_quantity[h], 'node_%s_%s' % (h, j))
                constraints_names.append(None)
            elif var_demand_node_pos[j, h] == 1:  # intermediate
                m.addConstr(flow_in_j == flow_out_j, 'node_%s_%s' % (h, j))
                constraints_names.append(None)

    pi_id_start, pi_id_end = id_useful_constraints(constraints_names)
    m.setObjective(quicksum(flow_quantity[i] for i, _ in enumerate(demand_edges)), GRB.MAXIMIZE)
    print("OPTIMIZING...")
    m.update()
    m.optimize()
    print("DONE, RESULT:", GUROBI_STATUS[m.status])
    return m, (pi_id_start, pi_id_end), constraints_names


def id_useful_constraints(constraints_names):
    """ [None, None A, ..., B, None, None] index of A and B"""
    id_start, id_end = None, None
    for i, con in enumerate(constraints_names):
        if con is not None:
            id_start = i
            break

    for i, con in enumerate(reversed(constraints_names)):
        if con is not None:
            id_end = i
            break

    return id_start, id_end


def edge_from_shadow_price(G, m, pi_index, constrs):
    st, en = pi_index
    shadow_price = m.getAttr(GRB.Attr.Pi)[st:en]
    idx = np.argmax(shadow_price)

    e1, e2 = constrs[idx][1]
    e1, e2 = make_existing_edge(G, e1, e2)
    return e1, e2


def isr_srt(G):
    """ ISR policy assumes that weights on the graph are updated """
    nodes, edges, paths = set(), set(), list()
    SG = get_supply_graph(G)
    for ed in get_demand_edges(G, is_check_unsatisfied=True):
        path = nx.shortest_path(SG, ed[0], ed[1], weight=co.ElemAttr.WEIGHT.value)
        is_unk_bro_path = False
        for nd in path:
            if G.nodes[nd][co.ElemAttr.POSTERIOR_BROKEN.value] > 0:
                nodes.add(nd)
                is_unk_bro_path = True

        for i in range(len(path)-1):
            n1, n2 = path[i], path[i+1]
            n1, n2 = make_existing_edge(G, n1, n2)
            if G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value] > 0:
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


def isr_pruning_demand(G):
    # set the infinite weight for the 0 capacity edges
    SG = get_supply_graph(G)
    quantity = 0
    for n1, n2, _ in SG.edges:
        cap = SG.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value]
        if cap <= 0:
            SG.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.WEIGHT.value] = np.inf

    for d1, d2, _ in get_demand_edges(G, is_check_unsatisfied=True):
        path = nx.shortest_path(SG, d1, d2, weight=co.ElemAttr.WEIGHT.value)
        if is_path_working(G, path):
            quantity += do_prune(G, path)
            discover_path(G, path, co.NodeState.WORKING.value)
    return quantity


def remaining_demand_endpoints(G, d_edges):
    cap = 0
    for e1, e2 in d_edges:
        e1, e2 = make_existing_edge(G, e1, e2)
        cap += G.edges[e1, e2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value]
    return cap


def run(config):
    stats_list = []

    # read graph and print stats
    G, elements_val_id, elements_id_val = init_graph(co.PATH_TO_GRAPH, config.graph_path, config.supply_capacity, config)
    print_graph_info(G)

    # normalize coordinates and break components
    dim_ratio = scale_coordinates(G)

    distribution, broken_nodes, broken_edges, perc_broken_elements = destroy(G, config.destruction_type, config.destruction_precision, dim_ratio,
                                                                             config.destruction_width, config.n_destruction, config.graph_dataset, config.seed, ratio=config.destruction_quantity,
                                                                             config=config)

    # add_demand_endpoints
    if config.is_demand_clique:
        add_demand_clique(G, config.n_demand_clique, config.demand_capacity, config)
    else:
        add_demand_pairs(G, config.n_demand_pairs, config.demand_capacity, config)

    # hypothetical routability
    if not is_feasible(G, is_fake_fixed=True):
        print("This instance is not solvable. Check the number of demand edges, theirs and supply links capacity.\n\n\n")
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

    # if config.monitoring_type == co.PriorKnowledge.FULL:
    #     gain_knowledge_all(G)

    assert config.monitors_budget == -1 or config.monitors_budget >= len(get_demand_nodes(G)), \
        "budget is {}, demand nodes are {}".format(config.monitors_budget, len(get_demand_nodes(G)))

    if config.monitors_budget == -1:  # -1 budget means to set automatically as get_demand_nodes(G)
        config.monitors_budget = get_demand_nodes(G)

    # start of the protocol
    while len(get_demand_edges(G, is_check_unsatisfied=True)) > 0:
        # go on if there are demand edges to satisfy, and still is_feasible

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
                 "packet_monitoring": packet_monitor}

        # START
        # PRUNING DEMAND AS ISR
        update_graph_probabilities(G)
        quantity = isr_pruning_demand(G)
        routed_flow += quantity
        stats["flow"] = routed_flow

        # begin GET DATA FOR OPTIMUM
        SG_edges = get_supply_graph(G).edges
        demand_edges = get_demand_edges(G, is_check_unsatisfied=True, is_residual=True)
        broken_supply_nodes = get_element_by_state_KT(G, co.GraphElement.NODE, co.NodeState.BROKEN, co.Knowledge.KNOW)
        unk_supply_nodes = get_element_by_state_KT(G, co.GraphElement.NODE, co.NodeState.UNK, co.Knowledge.KNOW)
        broken_supply_edges = get_element_by_state_KT(G, co.GraphElement.EDGE, co.NodeState.BROKEN, co.Knowledge.KNOW)
        unk_supply_edges = get_element_by_state_KT(G, co.GraphElement.EDGE, co.NodeState.UNK, co.Knowledge.KNOW)
        bro_unk_node = list(set(unk_supply_nodes).union(set(broken_supply_nodes)))
        bro_unk_edge = list(set(unk_supply_edges).union(set(broken_supply_edges)))
        # end GET DATA FOR OPTIMUM

        # begin CORE
        m, pi_index, constrs = shp_optimum(G, demand_edges, G.nodes, broken_supply_edges, SG_edges, bro_unk_node, bro_unk_edge)
        ed1, ed2 = edge_from_shadow_price(G, m, pi_index, constrs)
        # begin CORE

        rep_nodes, rep_edges = [], []
        rep_b = do_repair_edge(G, ed1, ed2)
        rep_c = do_repair_node(G, ed1)
        rep_d = do_repair_node(G, ed2)
        rep_nodes.append(rep_c)
        rep_nodes.append(rep_d)
        rep_edges.append(rep_b)

        rep_nodes = [rp for rp in rep_nodes if rp is not None]
        rep_edges = [rp for rp in rep_edges if rp is not None]

        print(rep_nodes, rep_edges)
        stats["edge"] += rep_edges
        stats["node"] += rep_nodes

        res_demand_edges = gu.get_demand_edges(G, is_check_unsatisfied=True)

        print("These are the residual demand edges:")
        print(len(res_demand_edges), res_demand_edges)
        stats_list.append(stats)

    return stats_list
