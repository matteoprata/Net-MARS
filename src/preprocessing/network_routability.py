import networkx as nx

import src.constants as co
import src.preprocessing.network_utils as gru
from gurobipy import *


def is_feasible(G, is_fake_fixed=True):
    """ System of equations checks if solution exists. """

    demand_edges = gru.get_demand_edges(G, is_check_unsatisfied=True, is_residual=True)
    supply_edges = gru.get_supply_edges(G)

    for n1, n2, f in demand_edges:  # there must exist at least a path in the graph between the demand endpoints
        if not nx.has_path(gru.get_supply_graph(G), n1, n2):
            print("No path existing in the supply graph between demand endpoints", (n1, n2))
            return False

    m = system_for_routability(G, demand_edges, supply_edges, None, is_fake_fixed)
    is_solution_ok = m.status == GRB.status.OPTIMAL

    # print("> System says it " + ("IS" if is_solution_ok else "IS NOT") + " routable")
    return is_solution_ok


# DO NOT USE THIS
def is_routable(G, knowledge, is_fake_fixed=False):
    """ Returns True if the system of linear equations and inequalities has at least one solution. """

    demand_edges = gru.get_demand_edges(G, is_check_unsatisfied=True, is_residual=True)
    demand_nodes = gru.get_demand_nodes(G)
    supply_edges = gru.get_supply_edges(G)

    if len(demand_edges) == 0:
        print("> No demand edge left.")
        return True
    else:
        if not is_fake_fixed:
            return False

    for node in demand_nodes:
        if gru.get_node_degree_working_edges(G, node, is_fake_fixed) <= 0:
            print("> Demand node is isolated in the supply graph.")
            return False

    # end preliminary checks
    m = system_for_routability(G, demand_edges, supply_edges, knowledge, is_fake_fixed)
    is_solution_ok = m.status == GRB.status.OPTIMAL

    # print("> System says it " + ("IS" if is_solution_ok else "IS NOT") + " routable")
    return is_solution_ok


def system_for_routability(G, demand_edges, supply_edges, knowledge, is_fake_fixed):
    """ Linear system of equations to check the routability. """

    var_demand_flows = []
    for i, (n1, n2, f) in enumerate(demand_edges):
        name_flow = 'F{}'.format(i)
        var_demand_flows.append((name_flow, f))

    # for endpoint source 0, mid 1, destination 2
    var_demand_node_pos = gru.demand_node_position(demand_edges, [name_flow for name_flow, _ in var_demand_flows], G.nodes)

    # edge: capacity map
    var_capacity_grid = gru.get_capacity_grid(G, knowledge, is_fake_fixed)

    ###################################################################################################################

    m = Model('netflow')

    m.setObjective(1, GRB.MAXIMIZE)
    m.params.OutputFlag = 0
    m.params.LogToConsole = 0

    # 1. create: flow variables f_ij^h
    flow_var = {}
    for h, _ in var_demand_flows:
        for i, j, _ in supply_edges:
            flow_var[h, i, j] = m.addVar(lb=0, ub=var_capacity_grid[i, j], vtype=GRB.CONTINUOUS, name='flow_var_{}_{}_{}'.format(h, i, j))
            flow_var[h, j, i] = m.addVar(lb=0, ub=var_capacity_grid[i, j], vtype=GRB.CONTINUOUS, name='flow_var_{}_{}_{}'.format(h, i, j))

    # 1. add: edge capacity constraints
    for i, j, _ in supply_edges:
        m.addConstr(quicksum(flow_var[h, i, j] + flow_var[h, j, i] for h, _ in var_demand_flows) <= var_capacity_grid[i, j], 'cap_%s_%s' % (i, j))

    # 2 add: flow conservation constraints
    for h, dem_val in var_demand_flows:
        for j in G.nodes:

            to_j, from_j = gru.get_incident_edges_of_node(node=j, edges=supply_edges)

            flow_out_j = quicksum(flow_var[h, j, k] for _, k in from_j)  # out flow da j
            flow_in_j = quicksum(flow_var[h, k, j] for k, _ in to_j)     # inner flow da j

            if var_demand_node_pos[j, h] == 0:    # source
                m.addConstr(flow_out_j - flow_in_j == dem_val, 'node_%s_%s' % (h, j))
            elif var_demand_node_pos[j, h] == 2:  # destination
                m.addConstr(flow_in_j - flow_out_j == dem_val, 'node_%s_%s' % (h, j))
            elif var_demand_node_pos[j, h] == 1:  # intermediate
                m.addConstr(flow_in_j == flow_out_j, 'node_%s_%s' % (h, j))

    m.update()
    m.optimize()
    return m


