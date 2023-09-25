import time

from src.preprocessing.network_init import *
from src.preprocessing.network_monitoring import *
from src.preprocessing.network_utils import *
import src.constants as co
import src.plotting.graph_plotting as pg

import src.preprocessing.network_utils as gru
from gurobipy import *
from src.recovery_protocols.RecoveryProtocol import RecoveryProtocol
import matplotlib.pyplot as plt


class MinTDS(RecoveryProtocol):
    file_name = "MinTDS"
    plot_name = "MinTDS"

    plot_marker = 0
    plot_color_curve = 9

    def __init__(self, config):
        super().__init__(config)
        self.N_STEPS = 50

    @staticmethod
    def run_header(config):
        stats_list = []

        # read graph and print stats
        G, elements_val_id, elements_id_val = init_graph(co.PATH_TO_GRAPH, config.graph_path, config.supply_capacity, config)
        print_graph_info(G)

        # normalize coordinates and break components
        dim_ratio = scale_coordinates(G)

        distribution, broken_nodes, broken_edges, perc_broken_elements = destroy(G, config.destruction_type,
                                                                                 config.destruction_precision,
                                                                                 dim_ratio,
                                                                                 config.destruction_width,
                                                                                 config.n_destruction,
                                                                                 config.graph_dataset,
                                                                                 config.seed,
                                                                                 ratio=config.destruction_quantity,
                                                                                 config=config)

        # add_demand_endpoints
        if config.is_demand_clique:
            add_demand_clique(G, config)
        else:
            add_demand_pairs(G, config.n_edges_demand, config.demand_capacity, config)

        # hypothetical routability
        if not is_feasible(G, is_fake_fixed=True):
            print( "This instance is not solvable. ",
                   "Check the number of demand edges, theirs and supply links capacity, or graph connectivity.\n\n\n")
            return

        # repair demand edges
        demand_node = get_demand_nodes(G)
        for dn in demand_node:
            do_repair_node(G, dn)
            # INITIAL NODES repairs are not counted in the stats

        iter = 0
        # true ruotability

        routed_flow = 0
        monitors_stats = set()
        demands_sat = {d: [] for d in get_demand_edges(G, is_capacity=False)}  # d1: [0, 1, 1, 0, 10] // demands_sat[d].append(0)

        # add monitors
        packet_monitor = 0
        for n1, n2, _ in get_demand_edges(G):
            G.nodes[n1][co.ElemAttr.IS_MONITOR.value] = True
            G.nodes[n2][co.ElemAttr.IS_MONITOR.value] = True
            monitors_stats |= {n1, n2}
            packet_monitor += do_k_monitoring(G, n1, config.k_hop_monitoring)
            packet_monitor += do_k_monitoring(G, n2, config.k_hop_monitoring)

        config.monitors_budget_residual -= len(monitors_stats)
        print("DEMAND EDGES", get_demand_edges(G))
        print("DEMAND NODES", get_demand_nodes(G))

        pg.plot(G, config.graph_path, distribution, config.destruction_precision, dim_ratio,
                config.destruction_show_plot, config.destruction_save_plot, config.seed, "TRU", co.PlotType.TRU,
                config.destruction_quantity)

        return G, stats_list, monitors_stats, packet_monitor, demands_sat, routed_flow, iter

    def run(self):
        comp_time_st = time.time()
        G, stats_list, monitors_stats, packet_monitor, demands_sat, routed_flow, iter = self.run_header(self.config)
        # print(len(G.nodes), "ed", len(gru.get_supply_edges(G)))
        m, force_repair = None, False

        # SOLVE LP
        broken_supply_edges = get_element_by_state_KT(G, co.GraphElement.EDGE, co.NodeState.BROKEN, co.Knowledge.TRUTH)
        broken_supply_nodes = get_element_by_state_KT(G, co.GraphElement.NODE, co.NodeState.BROKEN, co.Knowledge.TRUTH)
        supply_edges = get_supply_graph(G).edges
        supply_nodes = G.nodes
        demand_edges = get_demand_edges(G, is_check_unsatisfied=False, is_residual=False)

        print("broken_supply_edges", broken_supply_edges)
        print("broken_supply_nodes", broken_supply_nodes)

        # SOLUTION OF THE MILP
        model = self.minTDS_opt(G, demand_edges, supply_edges, supply_nodes, broken_supply_edges, broken_supply_nodes)

        # SETUP VAR
        node_rep, edge_rep, total_flow, demands_sat = self.interpret_opt_var(model, G, broken_supply_nodes, broken_supply_edges, demand_edges)

        # packet_monitor -- monitors paced up to iteration i
        # monitors -- monitors placed up to now (no duplicates)
        comp_time_et = time.time()
        elapsed_time = comp_time_et - comp_time_st  # elapsed time in seconds

        stats = {"iter": 0,
                 "node": node_rep,
                 "edge": edge_rep,
                 "flow": total_flow,
                 "monitors": monitors_stats,
                 "packet_monitoring": packet_monitor,
                 "demands_sat": demands_sat,
                 "execution_sec": elapsed_time
                 }
        return [stats]

    def minTDS_opt(self, G, demand_edges, supply_edges, supply_nodes, broken_supply_edges, broken_supply_nodes):
        # demand edges // supply edges // nodes // nodes (broken, unk) // supply edges (broken) // supply edges (broken, unk)
        var_demand_flows = [(i, f) for i, (_, _, f) in enumerate(demand_edges)]   # [(d1, f1), ...]

        # for endpoint source 0, mid 1, destination 2
        # {(n, d): position, }
        var_demand_node_pos = gru.demand_node_position(demand_edges, [name_flow for name_flow, _ in var_demand_flows], G.nodes)

        ###################################################################################################################

        m = Model('netflow')

        m.setParam('TimeLimit', 5 * 60)
        m.params.OutputFlag = 0
        m.params.LogToConsole = 0

        # 1. create: flow variables f_ijh
        flow_var = {}
        N = self.N_STEPS   # sum the number of node and edges
        for n in range(N):
            for h, dem_val in var_demand_flows:
                for i, j, _ in supply_edges:
                    flow_var[h, i, j, n] = m.addVar(lb=0, ub=G.edges[i, j, co.EdgeType.SUPPLY.value][co.ElemAttr.CAPACITY.value],
                                                    vtype=GRB.CONTINUOUS, name='flow_var_{}_{}_{}_{}'.format(h, i, j, n))

                    flow_var[h, j, i, n] = m.addVar(lb=0, ub=G.edges[i, j, co.EdgeType.SUPPLY.value][co.ElemAttr.CAPACITY.value],
                                                    vtype=GRB.CONTINUOUS, name='flow_var_{}_{}_{}_{}'.format(h, j, i, n))

        # 2. create: perc flow variable a_i(n), g_i(n)
        alpha_var, gamma_var = {}, {}
        for n in range(N):
            for h, _ in var_demand_flows:
                alpha_var[h, n] = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='alpha_var_{}_{}'.format(h, n))
                gamma_var[h, n] = m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='gamma_var_{}_{}'.format(h, n))  # 1 iff a == 1

        # 3. create: repair edge variable y_ij(n)
        rep_edge_var = {}
        for n in range(N):
            for n1, n2, _ in supply_edges:
                rep_edge_var[n1, n2, n] = m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='rep_edge_var_{}_{}_{}'.format(n1, n2, n))

        # 4. create: repair node variable y_i(n)
        rep_node_var = {}
        for n in range(N):
            for n1 in supply_nodes:
                rep_node_var[n1, n] = m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='rep_node_var_{}_{}'.format(n1, n))

        for n1, n2, _ in supply_edges:
            m.addConstr(quicksum(rep_edge_var[n1, n2, n] for n in range(N)) <= 1)

        for n1 in supply_nodes:
            m.addConstr(quicksum(rep_node_var[n1, n] for n in range(N)) <= 1)

        # CONST
        for i, j, _ in supply_edges:
            n1bro = G.nodes[i][co.ElemAttr.STATE_TRUTH.value]
            n2bro = G.nodes[j][co.ElemAttr.STATE_TRUTH.value]
            ebro = G.edges[i, j, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value]

            if ebro + n1bro + n2bro == 0:  # WORKING at time 0
                m.addConstr(rep_edge_var[i, j, 0] == 1)
            else:
                m.addConstr(rep_edge_var[i, j, 0] == 0)

            if n1bro == 0:  # WORKING at time 0
                m.addConstr(rep_node_var[i, 0] == 1)
            else:
                m.addConstr(rep_node_var[i, 0] == 0)

            if n2bro == 0:  # WORKING at time 0
                m.addConstr(rep_node_var[j, 0] == 1)
            else:
                m.addConstr(rep_node_var[j, 0] == 0)

        # CONSTRAINTS
        BUDGET_REP = 1
        for n in range(N):  # reps non superino il budget
            m.addConstr(quicksum(rep_edge_var[i, j, n] for i, j, _ in broken_supply_edges) +
                        quicksum(rep_node_var[i, n] for i in broken_supply_nodes)
                        <= BUDGET_REP)

        print(broken_supply_edges)
        print(broken_supply_nodes)

        for n in range(N):
            for h, _ in var_demand_flows:
                m.addConstr(gamma_var[h, n] >= 1 - alpha_var[h, n])
                if n > 0:
                    m.addConstr(gamma_var[h, n-1] >= gamma_var[h, n])

        for i, j, _ in supply_edges:
            for ngen in range(N):
                was_ever_repaired = quicksum(rep_edge_var[i, j, n] for n in range(ngen))
                # A
                m.addConstr(quicksum(flow_var[h, i, j, ngen] + flow_var[h, j, i, ngen] for h, _ in var_demand_flows) <=
                            G.edges[i, j, co.EdgeType.SUPPLY.value][co.ElemAttr.CAPACITY.value] * was_ever_repaired,
                            'qcap_{}_{}'.format(i, j))

                # # B1
                # m.addConstr(rep_edge_var[i, j, ngen] <= quicksum(rep_node_var[i, n] for n in range(ngen)),
                #             'wcap_{}_{}'.format(i, j))
                # # B2
                # m.addConstr(rep_edge_var[i, j, ngen] <= quicksum(rep_node_var[j, n] for n in range(ngen)),
                #             'ecap_{}_{}'.format(i, j))

        # tutti gli alrchi possono essere iparati al massimo in uno step
        for i, j, _ in supply_edges:
            m.addConstr(quicksum(rep_edge_var[i, j, n] for n in range(N)) <= 1)

        for n1 in supply_nodes:
            m.addConstr(quicksum(rep_node_var[n1, n] for n in range(N)) <= 1)

        # 2 add: flow conservation constraints C
        MAX_FLOW = len(demand_edges) * 100  # CAREFUL 100 is the max flow per pair?
        for n in range(N):
            for h, dem_val in var_demand_flows:  # [(d1, f1)]
                for j in G.nodes:
                    to_j, from_j = gru.get_incident_edges_of_node(node=j, edges=supply_edges)  # to_j [(1,600), (50, 600)], [(600, 1), (600, 50)]

                    flow_out_j = quicksum(flow_var[h, j, k, n] for _, k in from_j)  # out flow da j
                    flow_in_j = quicksum(flow_var[h, k, j, n] for k, _ in to_j)     # inner flow da j

                    if var_demand_node_pos[j, h] == 0:    # source
                        m.addConstr(flow_out_j - flow_in_j == dem_val * alpha_var[h, n], 'node_%s_%s' % (h, j))

                    elif var_demand_node_pos[j, h] == 2:  # destination
                        m.addConstr(flow_in_j - flow_out_j == dem_val * alpha_var[h, n], 'node_%s_%s' % (h, j))

                    elif var_demand_node_pos[j, h] == 1:  # intermediate
                        m.addConstr(flow_in_j == flow_out_j, 'node_%s_%s' % (h, j))

                    was_ever_repaired = quicksum(rep_node_var[j, ni] for ni in range(n))
                    m.addConstr(flow_in_j + flow_out_j <= was_ever_repaired * MAX_FLOW, "quest")

        # OBJECTIVE
        eps = .0001
        rep_shit = quicksum(rep_edge_var[i, j, n] for i, j, _ in broken_supply_edges for n in range(N)) + \
                   quicksum(rep_node_var[n1, n] for n1 in broken_supply_nodes for n in range(N))

        obj1 = quicksum(gamma_var[h, n] + eps * rep_shit for h, _ in var_demand_flows for n in range(N))
        obj2 = quicksum(alpha_var[h, n] * flow for h, flow in var_demand_flows for n in range(N))
        m.setObjective(obj1, GRB.MINIMIZE)

        print("OPTIMIZING...")
        m.update()
        m.optimize()
        print("DONE, RESULT:", co.GUROBI_STATUS[m.status])
        # debug(model, G, var_demand_flows, broken_unk_edges)
        return m

    def interpret_opt_var(self, m, G, broken_supply_nodes, broken_supply_edges, demand_edges):
        var_demand_flows = [(i, f) for i, (_, _, f) in enumerate(demand_edges)]
        node_rep, edge_rep = [], []
        if m.status == GRB.status.OPTIMAL:
            for n in range(self.N_STEPS):
                # ELEMENTS TO REPAIR
                for i, j, _ in broken_supply_edges:
                    ed_val = int(m.getVarByName('rep_edge_var_{}_{}_{}'.format(i, j, n)).x)
                    if ed_val > 0:
                        print("time", n, "rep", i, j)
                        edge_rep.append((i, j))

                for i in broken_supply_nodes:
                    no_val = int(m.getVarByName('rep_node_var_{}_{}'.format(i, n)).x)
                    if no_val > 0:
                        print("time", n, "rep node", i)
                        node_rep.append(i)

            # TIMES TO FLOW RESTORED
            n_to_route = np.ones(shape=(len(var_demand_flows))) * -1   # [15, 6, 2] times of total demand restoration
            for i, (h, _) in enumerate(var_demand_flows):
                for n in range(self.N_STEPS):
                    var = m.getVarByName('alpha_var_{}_{}'.format(h, n)).x
                    if var == 1:
                        n_to_route[i] = n
                        break

            if -1 in n_to_route:
                print("Careful, this instance was not solved correctly. One demand was not satisfied.")
                exit()

            last_time = int(max(n_to_route))
            flow_vec = np.zeros(shape=(last_time, len(var_demand_flows)), dtype=np.int8)  # [000000000]
            for i, (h, _) in enumerate(var_demand_flows):
                when_total_flow_h = int(n_to_route[i])-1
                flow_vec[when_total_flow_h, i] = self.config.demand_capacity   # [00000 30 0000]

            total_flow = np.sum(flow_vec)
            demand_flows = {(a, b): list(flow_vec[:, i]) for i, (a, b, _) in enumerate(demand_edges)}  # dict

        return node_rep, edge_rep, total_flow, demand_flows

