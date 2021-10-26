import src.constants as co
from graph_preprocessing import *
from gurobipy import *


def is_routable(H, green_edges):
    prep = prepare_for_optimization(H, green_edges)

    if type(prep) == bool:
        return prep
    else:
        nodes, demand_flows, arcs, capacity, demand_node_pos = prep

    m = system_for_routability(nodes, demand_flows, arcs, capacity, demand_node_pos)
    result = check_model_solution(m)
    return result


def demand_node_position(demand_edges, demands, nodes):
    """ maps every node to their position in the graph wrt the demand graph """
    demand_node_pos = {}

    for node in nodes:
        for h in demands:
            demand_node_pos[node, h] = 1  # 1 nodo centrale di demand

    sources = [i for i, _, _ in demand_edges]
    targets = [j for _, j, _ in demand_edges]

    for i, h in enumerate(demands):
        demand_node_pos[sources[i], h] = 0  # 0 nodo sorgente di demand
        demand_node_pos[targets[i], h] = 2  # 2 nodo target di demand

    return demand_node_pos

def prepare_for_optimization(G):
    """ Qui il grafo in input e' di demand, non di supply... Returns either bool if the answer of opt is trivial or a tuple."""

    demand_edges = get_demand_edges(G)
    demand_nodes = get_demand_nodes(G)

    if len(demand_edges) == 0:
        print("> No demand edge left.")
        return True

    for node in demand_nodes:
        if get_node_degree(G, node) <= 0:
            print("> Demand node is isolated in the supply graph.")
            return False

    dvar_demand_flows = []
    for i, n1, n2, f in enumerate(dvar_demand_flows):
        name_flow = 'F{}'.format(i)
        dvar_demand_flows.append((name_flow, f))

    # 0 source 1 central 2 target
    demand_node_pos = demand_node_position(demand_edges, [name_flow for name_flow, _ in dvar_demand_flows], G.nodes)

    return nodes, demand_flows, arcs, capacity, demand_node_pos


def demand_node_position(demand_edges, demands, nodes):
    """ maps every node to their position in the graph wrt the demand graph """
    demand_node_pos = {}

    for node in nodes:
        for h in demands:
            demand_node_pos[node, h] = 1  # 1 nodo centrale di demand

    sources = [i for i, _, _ in demand_edges]
    targets = [j for _, j, _ in demand_edges]

    for i, h in enumerate(demands):
        demand_node_pos[sources[i], h] = 0  # 0 nodo sorgente di demand
        demand_node_pos[targets[i], h] = 2  # 2 nodo target di demand

    return demand_node_pos


def system_for_routability(nodes, demand_flows, arcs, capacity, demand_node_pos):
    """ Returns True if the system of linear equations and inequalities has at least one solution. """

    m = Model('netflow')

    m.setObjective(1, GRB.MAXIMIZE)
    m.params.OutputFlag = 0

    # 1. CREA: variabili decisionali di flusso f_ij^h(n)
    flow_var = {}
    for h, _ in demand_flows:
        for i, j in arcs:
            flow_var[h, i, j] = m.addVar(lb=0, ub=capacity[i, j], vtype=GRB.CONTINUOUS, name='flow_var_%s_%s_%s' % (h, i, j))
            flow_var[h, j, i] = m.addVar(lb=0, ub=capacity[i, j], vtype=GRB.CONTINUOUS, name='flow_var_%s_%s_%s' % (h, j, i))

    # 1. ADD: Arc capacity constraints
    for i, j in arcs:
        m.addConstr(quicksum(flow_var[h, i, j] + flow_var[h, j, i] for h, _ in demand_flows) <= capacity[i, j],'cap_%s_%s' % (i, j))

    # 2 ADD: Flow conservation constraints
    for h, dem_val in demand_flows:
        for j in nodes:

            to_j, from_j = get_incident_edges_of_node(node=j, edges=arcs)

            flow_out_j = quicksum(flow_var[h, j, k] for _, k in from_j)  # flusso uscente da j
            flow_in_j = quicksum(flow_var[h, k, j] for k, _ in to_j)  # flusso entrante da j

            # demand_node_pos[j, h] = 0 se h e' un nodo sorgente per la domanda h, 1 intermedio, 2 target, 0 source
            if demand_node_pos[j, h] == 0:  # il nodo non ha archi entrati, e' dunque un nodo sorgente
                m.addConstr(flow_out_j - flow_in_j == dem_val, 'node_%s_%s' % (h, j))

            elif demand_node_pos[j, h] == 2:  # il nodo non ha archi uscenti, e' dunque un nodo pozzo
                m.addConstr(flow_in_j - flow_out_j == dem_val, 'node_%s_%s' % (h, j))

            elif demand_node_pos[j, h] == 1:
                m.addConstr(flow_in_j == flow_out_j, 'node_%s_%s' % (h, j))

    m.update()

    # Compute optimal solution
    m.optimize()
    return m


def check_model_solution(m):
    status = m.status == GRB.status.OPTIMAL

    # se i vincoli sono rispettati torna TRUE
    if status:
        print("CHECK routability: True")
    else:
        print("CHECK routability: False")

    return status