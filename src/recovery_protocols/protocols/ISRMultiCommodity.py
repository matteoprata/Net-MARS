
from src.preprocessing.network_init import *
from src.preprocessing.network_monitoring import *
from src.preprocessing.network_utils import *
import src.constants as co

import src.utilities.util_routing_stpath as mxv

import src.preprocessing.network_utils as gru
from gurobipy import *

from src.recovery_protocols.protocols.ISR import ISR


class ISRMultiCommodity(ISR):

    file_name = "ISRMult"
    plot_name = "ISR-Mult"

    plot_marker = "<"
    plot_color_curve = 6

    def __init__(self, config):
        super().__init__(config)

    def run(self):
        G, stats_list, monitors_stats, packet_monitor, demands_sat, routed_flow, iter = self.prepare_run()
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

            # PRUNING A LA IP
            if self.config.is_IP_routing:
                for d1, d2, _ in get_demand_edges(G, is_check_unsatisfied=True):
                    SG = get_supply_graph(G)
                    path_prune, _, _, is_working = mxv.protocol_routing_IP(SG, d1, d2)
                    if is_working:
                        quantity_pruning = do_prune(G, path_prune)
                        routed_flow += quantity_pruning
                        d_edge = make_existing_edge(path_prune[0], path_prune[-1])
                        demand_edges_routed_flow_pp[d_edge] += quantity_pruning
                        stats["flow"] = routed_flow
                        print("pruned", quantity_pruning, "on", path_prune)
                self.demand_log(G, demands_sat, stats, self.config, is_monotonous=False)
            else:
                # PRUNING
                quantity, rep_nodes, rep_edges = self.flow_var_pruning_demand(G, m, force_repair, demand_edges_routed_flow_pp)
                stats["edge"] += rep_edges
                stats["node"] += rep_nodes
                routed_flow += quantity
                stats["flow"] = routed_flow

                TOTAL_FLOW = self.config.demand_capacity * self.config.n_edges_demand
                # no repair was done and no total flow routed
                self.demand_log(G, demands_sat, stats, self.config, is_monotonous=False)

                print("DEB", force_repair, len(rep_edges) + len(rep_nodes), quantity, TOTAL_FLOW)
                if force_repair and len(rep_edges) + len(rep_nodes) == 0 and quantity != TOTAL_FLOW:
                    print("Cap-ISOLATED node. Infeasibility.")
                    stats_list.append(stats)
                    return stats_list

            res_demand_edges = gu.get_demand_edges(G, is_check_unsatisfied=True)
            print("PERO HO", res_demand_edges)
            reset_supply_edges(G)

            print("These are the residual demand edges:")
            print(len(res_demand_edges), res_demand_edges)

            if len(res_demand_edges) == 0:
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
            opt = self.relaxed_LP_multicom(G, demand_edges, G.nodes, broken_supply_edges, SG_edges, bro_unk_node, bro_unk_edge)
            var_demand_flows, var_demand_node_pos, supply_edges, m = opt

            # vv = get_element_by_state_KT(G, co.GraphElement.NODE, co.NodeState.DESTROY, co.Knowledge.TRUTH)
            # print("broken", len(vv), vv)

            node_rep, path_nodes, path_edges = self.derive_solution_from_variables(G, var_demand_flows, var_demand_node_pos, supply_edges, m)
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
                if len(res_demand_edges) > 0 and self.config.monitors_budget_residual > 0:
                    G.nodes[node_rep][co.ElemAttr.IS_MONITOR.value] = True
                    monitors_stats |= {node_rep}
                    stats["monitors"] |= monitors_stats
                    self.config.monitors_budget_residual -= 1

                # k-discovery
                packet_monitor += k_hop_discovery(G, node_rep, self.config.k_hop_monitoring)
                stats["packet_monitoring"] = packet_monitor

            stats_list.append(stats)
        return stats_list

    @staticmethod
    def relaxed_LP_multicom(G, demand_edges, supply_nodes, broken_supply_edges, supply_edges, broken_unk_nodes,
                            broken_unk_edges):
        # demand edges // supply edges // nodes // nodes (broken, unk) // supply edges (broken) // supply edges (broken, unk)
        var_demand_flows = [(i, f) for i, (_, _, f) in enumerate(demand_edges)]  # [(d1, f1), ...]

        # for endpoint source 0, mid 1, destination 2
        # {(n, d): position, }
        var_demand_node_pos = gru.demand_node_position(demand_edges, [name_flow for name_flow, _ in var_demand_flows],
                                                       G.nodes)

        ###################################################################################################################

        m = Model('netflow')

        # model.setObjective(1, GRB.MAXIMIZE)
        m.params.OutputFlag = 0
        m.params.LogToConsole = 0

        # 1. create: flow variables f_ij^h
        flow_var = {}
        for h, dem_val in var_demand_flows:
            for i, j, _ in supply_edges:
                flow_var[h, i, j] = m.addVar(lb=0,
                                             ub=G.edges[i, j, co.EdgeType.SUPPLY.value][co.ElemAttr.CAPACITY.value],
                                             vtype=GRB.CONTINUOUS, name='flow_var_{}_{}_{}'.format(h, i, j))

                flow_var[h, j, i] = m.addVar(lb=0,
                                             ub=G.edges[i, j, co.EdgeType.SUPPLY.value][co.ElemAttr.CAPACITY.value],
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
        #     alpha_var[h] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='alpha_var_{}'.format(h))

        # print(len(supply_edges - broken_unk_edges), len(supply_edges), len(broken_unk_edges))
        for e in supply_edges:
            i, j, _ = e

            m.addConstr(quicksum(flow_var[h, i, j] + flow_var[h, j, i] for h, _ in var_demand_flows) <=
                        G.edges[i, j, co.EdgeType.SUPPLY.value][co.ElemAttr.CAPACITY.value] * rep_edge_var[i, j],
                        'wor_%s_%s' % (i, j))

        # for e0, e1, _ in supply_edges:  # repairing an edge requires reparing the nodes and viceversa
        #     model.addConstr(rep_edge_var[e0, e1] <= rep_node_var[e1])
        #     model.addConstr(rep_edge_var[e0, e1] <= rep_node_var[e0])

        # for e0, e1, _ in supply_edges:  # repairing an edge requires reparing the nodes and viceversa
        for i in supply_nodes:  # broken_supply_edges
            m.addConstr(rep_node_var[i] * net_max_degree(G) >= quicksum(
                rep_edge_var[e1, e2] for e1, e2, _ in broken_supply_edges if i in [e1, e2]))

        # 2 add: flow conservation constraints
        for h, dem_val in var_demand_flows:  # [(d1, f1)]
            for j in G.nodes:
                to_j, from_j = gru.get_incident_edges_of_node(node=j,
                                                              edges=supply_edges)  # to_j [(1,600), (50, 600)], [(600, 1), (600, 50)]

                flow_out_j = quicksum(flow_var[h, j, k] for _, k in from_j)  # out flow da j
                flow_in_j = quicksum(flow_var[h, k, j] for k, _ in to_j)  # inner flow da j

                if var_demand_node_pos[j, h] == 0:  # source
                    m.addConstr(flow_out_j - flow_in_j == dem_val, 'node_%s_%s' % (h, j))
                elif var_demand_node_pos[j, h] == 2:  # destination
                    m.addConstr(flow_in_j - flow_out_j == dem_val, 'node_%s_%s' % (h, j))
                elif var_demand_node_pos[j, h] == 1:  # intermediate
                    m.addConstr(flow_in_j == flow_out_j, 'node_%s_%s' % (h, j))

        # OBJECTIVE
        m.setObjective(quicksum(
            rep_node_var[ni] * co.REPAIR_COST * G.nodes[ni][co.ElemAttr.POSTERIOR_BROKEN.value] for ni in
            broken_unk_nodes) +
                       quicksum(rep_edge_var[n1, n2] * co.REPAIR_COST * G.edges[n1, n2, co.EdgeType.SUPPLY.value][
                           co.ElemAttr.POSTERIOR_BROKEN.value] for n1, n2, _ in broken_unk_edges),
                       GRB.MINIMIZE)

        print("OPTIMIZING...")
        m.update()
        m.optimize()
        print("DONE, RESULT:", co.GUROBI_STATUS[m.status])
        return var_demand_flows, var_demand_node_pos, supply_edges, m

    @staticmethod
    def derive_solution_from_variables(G, demand_flows, var_demand_node_pos, supply_edges, model):
        """ returns the node with maximum in flow """
        total_flow_node = defaultdict(int)  # all the nodes associated the flow
        path_nodes, path_edges = set(), set()

        if model.status == GRB.status.OPTIMAL:
            for h, _ in enumerate(demand_flows):
                for i, j, _ in supply_edges:

                    var_value_v1 = model.getVarByName(
                        'flow_var_{}_{}_{}'.format(h, i, j))  # edge variable to check if > 0
                    var_value_v2 = model.getVarByName(
                        'flow_var_{}_{}_{}'.format(h, j, i))  # edge variable to check if > 0

                    if var_value_v1.x > 0 or var_value_v2.x > 0:
                        # print("positive var", h, i, j, var_value_v1.x, var_value_v2.x, "pbro:", G.nodes[i][co.ElemAttr.POSTERIOR_BROKEN.value])

                        n1bro = G.nodes[i][co.ElemAttr.STATE_TRUTH.value]
                        n2bro = G.nodes[j][co.ElemAttr.STATE_TRUTH.value]
                        ebro = G.edges[i, j, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value]

                        if n1bro + n2bro + ebro > 0:  # broken
                            if var_demand_node_pos[i, h] == 0:  # in or out flow
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
        for k in total_flow_node:  # total_flow_node is empy when path nodes è vuoto o tutto funziona
            if total_flow_node[k] > max_flow:
                max_flow = total_flow_node[k]
                max_flow_node = k

        # print(max_flow_node, max_flow, path_nodes, total_flow_node)
        if max_flow_node is None:
            return None, path_nodes, path_edges

        # broken_flow_nodes = total_flow_node.keys()
        # print("broken to choose", max_flow_node, max_flow, broken_flow_nodes)
        return max_flow_node, path_nodes, path_edges

    @staticmethod
    def flow_var_pruning_demand(G, m, force_repair, demand_edges_routed_flow_pp):
        # set the infinite weight for the 0 capacity edges

        quantity = 0
        rep_nodes, rep_edges = set(), set()

        if m is None:
            return quantity, rep_nodes, rep_edges

        SGOut = get_supply_graph(G)
        for h, (d1, d2, _) in enumerate(
                get_demand_edges(G, is_check_unsatisfied=False, is_residual=False)):  # enumeration is coherent
            SG = nx.Graph()
            dem = G.edges[d1, d2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value]

            for i, j, _ in SGOut.edges:
                var_value_v1 = m.getVarByName('flow_var_{}_{}_{}'.format(h, i, j)).x  # edge variable to check if > 0
                var_value_v2 = m.getVarByName('flow_var_{}_{}_{}'.format(h, j, i)).x  # edge variable to check if > 0

                n1bro = G.nodes[i][co.ElemAttr.STATE_TRUTH.value]
                n2bro = G.nodes[j][co.ElemAttr.STATE_TRUTH.value]
                ebro = G.edges[i, j, co.EdgeType.SUPPLY.value][co.ElemAttr.STATE_TRUTH.value]

                if var_value_v1 > 0 or var_value_v2 > 0:
                    if n1bro + n2bro + ebro == 0 or force_repair:
                        # print("added", h, d1, d2, i, j, var_value_v1, var_value_v2, n1bro, n2bro, ebro)
                        # if force_repair:  # this because sometimes every node is ok, but some edges of working nodes are not repaired
                        repn, repe = do_repair_full_edge(G, i, j)
                        rep_nodes |= set(repn)
                        rep_edges |= set(repe)

                        na, nb = make_existing_edge(i, j)
                        cap = min(G.edges[na, nb, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value], dem)
                        # print(na, nb, cap, [d1, d2], (dem))
                        SG.add_edge(na, nb, capacity=cap)

            # poz = nx.spring_layout(SG)
            # nx.draw_networkx_edge_labels(SG, pos=poz)
            # nx.draw(SG, with_labels=True, pos=poz)
            # plt.show()

            if d1 in SG.nodes and d2 in SG.nodes and nx.has_path(SG, d1, d2):
                pruned_quant, flow_edge = nx.maximum_flow(SG, d1, d2)
                pruned_quant = 0 if pruned_quant < dem else min(dem, pruned_quant)
                demand_edges_routed_flow_pp[(d1, d2)] += pruned_quant
                quantity += pruned_quant
                G.edges[d1, d2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value] -= pruned_quant
                for n1 in flow_edge:
                    for n2 in flow_edge[n1]:
                        G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value] -= flow_edge[n1][
                            n2]
        return quantity, rep_nodes, rep_edges