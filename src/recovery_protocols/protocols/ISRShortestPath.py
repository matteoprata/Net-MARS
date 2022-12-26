
from src.preprocessing.graph_preprocessing import *
from src.preprocessing.graph_monitoring import *
from src.preprocessing.graph_utils import *
import src.constants as co

import src.utilities.util_routing_stpath as mxv
from src.recovery_protocols.protocols.ISR import ISR


class ISRShortestPath(ISR):
    file_name = "ISRSTP"
    plot_name = "ISR-STP"

    plot_marker = ">"
    plot_color_curve = 5

    def __init__(self, config):
        super().__init__(config)

    def run(self):
        G, stats_list, monitors_stats, packet_monitor, demands_sat, routed_flow, iter = self.prepare_run()

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

            self.update_graph_probabilities(G)

            # ROUTING A-LA IP
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
            else:
                quantity = self.isr_pruning_demand(G, demand_edges_routed_flow_pp)
                routed_flow += quantity
                stats["flow"] = routed_flow

            paths_nodes, paths_edges, paths_tot = self.isr_srt(G)
            paths_elements = paths_nodes.union(paths_edges)
            paths_nodes, paths_edges = list(paths_nodes), list(
                paths_edges)  # nodes for which the state is broken or unknown

            print(paths_nodes)
            print(paths_edges)

            if len(paths_elements) == 0:  # it may be all working elements, do not return!, rather go on
                print("Process completed!", get_residual_demand(G), get_demand_edges(G, True, True, True))
                self.demand_log(G, demands_sat, stats, self.config)
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

                    self.demand_log(G, demands_sat, stats, self.config)
                    stats_list.append(stats)
                    continue

                # find the best node to repair based on max flow
                r_rem_tot = get_residual_demand(G)
                values_of_v = []
                for v in paths_nodes:
                    _, endpoints = self.shortest_through_v(v, paths_tot)  # per MULTI commodity usare il nodo i che max sum_j fij + f(ji)
                    f_rem_v = self.remaining_demand_endpoints(G, endpoints)
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
                print("These are the residual demand edges:")
                print(len(res_demand_edges), res_demand_edges)

                # add monitor to v_rep
                if len(res_demand_edges) > 0 and self.config.monitors_budget_residual > 0:
                    G.nodes[node_rep][co.ElemAttr.IS_MONITOR.value] = True
                    monitors_stats |= {node_rep}
                    stats["monitors"] |= monitors_stats
                    self.config.monitors_budget_residual -= 1

                # k-discovery
                packet_monitor += k_hop_discovery(G, node_rep, self.config.k_hop_monitoring)
                stats["packet_monitoring"] = packet_monitor

            self.demand_log(G, demands_sat, stats, self.config)
            stats_list.append(stats)

        return stats_list

    @staticmethod
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

            for i in range(len(path) - 1):
                n1, n2 = path[i], path[i + 1]
                n1, n2 = make_existing_edge(n1, n2)
                if G.edges[n1, n2, co.EdgeType.SUPPLY.value][
                    co.ElemAttr.POSTERIOR_BROKEN.value] > 0:  # edges broken or unk
                    edges |= {(n1, n2)}
                    is_unk_bro_path = True

            if is_unk_bro_path:
                paths.append(path)
        return nodes, edges, paths

    @staticmethod
    def update_graph_probabilities(G):
        """ shortest path weights, probabilistic """
        for n1, n2, et in G.edges:
            if et == co.EdgeType.SUPPLY.value:
                pn1 = G.nodes[n1][co.ElemAttr.POSTERIOR_BROKEN.value]
                pn2 = G.nodes[n2][co.ElemAttr.POSTERIOR_BROKEN.value]
                pn1n2 = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.POSTERIOR_BROKEN.value]
                cost = co.REPAIR_COST * ((pn1 + pn2) / 2 + pn1n2)
                G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.WEIGHT.value] = cost

    @staticmethod
    def shortest_through_v(v, paths_tot):
        paths, endpoints = [], []
        for pa in paths_tot:
            if v in pa:
                paths.append(v)
                endpoints.append((pa[0], pa[-1]))
        return paths, endpoints

    @staticmethod
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

    @staticmethod
    def remaining_demand_endpoints(G, d_edges):
        cap = 0
        for e1, e2 in d_edges:
            e1, e2 = make_existing_edge(e1, e2)
            cap += G.edges[e1, e2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value]
        return cap
