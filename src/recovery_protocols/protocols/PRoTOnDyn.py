import numpy as np

import src.plotting.graph_plotting as pg
from src.preprocessing.network_init import *
from src.preprocessing.network_monitoring import *
from src.preprocessing.network_utils import *
import src.constants as co
import random
import time
from src.recovery_protocols.utils import finder_recovery_path as frp, finder_recovery_path_pick as frpp, \
    adding_monitors as mon

from src.recovery_protocols.RecoveryProtocol import RecoveryProtocol
from src.recovery_protocols.RecoveryLog import RecoveryLog

from enum import Enum


class PRoTOnDyn(RecoveryProtocol):

    file_name = "PRoTOnDyn"
    plot_name = "PRoTOn Dyn"

    mode_path_repairing = co.ProtocolRepairingPath.MIN_COST_BOT_CAP
    mode_path_choosing_repair = co.ProtocolPickingPath.MIN_COST_BOT_CAP
    mode_monitoring = co.ProtocolMonitorPlacement.BUDGET
    mode_monitoring_type = co.PriorKnowledge.TOMOGRAPHY

    plot_marker = "D"
    plot_color_curve = 4

    def __init__(self, config):
        super().__init__(config)
        self.seed_fix_max_flow = 0
        self.random_gen_fix_max_flow = np.random.RandomState(seed=self.seed_fix_max_flow)

        self.MISSION_DURATION = 500
        # Viviana: con un processo di Poisson decidiamo gli "arrival time" delle distruzioni dinamiche

        rate_failure = 1 / 50  # tempo medio fra due rotture dinamiche
        rate_new_node = 1 / 200  # tempo medio fra due rotture dinamiche 450
        rate_new_edge = 1 / 45000  # tempo medio fra due rotture dinamiche 450
        rate_remove_node = 1 / 45000 # tempo medio fra due rotture dinamiche
        rate_remove_edge = 1 / 45000  # tempo medio fra due rotture dinamiche

        num_arrivals = self.MISSION_DURATION   # numero di arrivi totali. Ne mettiamo uno alto per fare esperimenti lunghi a piacimento
        # ma non ci interessano tutti.
        self.first_event_time = 30

        self.destroy_times = np.array(self.poisson_process_dynamic_events(rate_failure, num_arrivals, self.first_event_time, self.config, 1))
        self.destroy_times = [175, 200, 225, 250, 275, 300, 325]  # self.destroy_times[self.destroy_times < (self.MISSION_DURATION - 15)]  # removes the last time steps

        self.new_node_times = np.array(self.poisson_process_dynamic_events(rate_new_node, num_arrivals, self.first_event_time, self.config, 2))
        self.new_node_times = [50, 100] # self.new_node_times[self.new_node_times < (self.MISSION_DURATION - 15)]  # removes the last time steps

        self.new_edge_times = np.array(self.poisson_process_dynamic_events(rate_new_edge, num_arrivals, self.first_event_time, self.config, 3))
        self.new_edge_times = [75, 125]  # self.new_edge_times[self.new_edge_times < (self.MISSION_DURATION - 15)]  # removes the last time steps

        self.remove_node_times = np.array(self.poisson_process_dynamic_events(rate_remove_node, num_arrivals, self.first_event_time, self.config, 4))
        self.remove_node_times = [400]  # self.remove_node_times[self.remove_node_times < (self.MISSION_DURATION - 15)]  # removes the last time steps

        self.remove_edge_times = np.array(self.poisson_process_dynamic_events(rate_remove_edge, num_arrivals, self.first_event_time, self.config, 5))
        self.remove_edge_times = [425]# self.remove_edge_times[self.remove_edge_times < (self.MISSION_DURATION - 15)]  # removes the last time steps

        print("This will be the events:")
        print(self.destroy_times)
        print(self.new_node_times)
        print(self.new_edge_times)
        print(self.remove_node_times)
        print(self.remove_edge_times)
        time.sleep(5)

    def run(self):
        rlog = RecoveryLog(self, self.config)

        # read graph and print stats
        G, elements_val_id, elements_id_val = init_graph(co.PATH_TO_GRAPH, self.config.graph_path, self.config.supply_capacity, self.config)
        print_graph_info(G)

        # normalize coordinates and break components
        dim_ratio = scale_coordinates(G)

        distribution, broken_nodes, broken_edges, perc_broken_elements = destroy(G, self.config.destruction_type, self.config.destruction_precision, dim_ratio,
                                                                                 self.config.destruction_width, self.config.n_destruction, self.config.graph_dataset, self.config.seed, ratio=self.config.destruction_quantity,
                                                                                 config=self.config)

        # add_demand_endpoints
        if self.config.is_demand_clique:
            add_demand_clique(G, self.config, self.random_gen_fix_max_flow)  # generators uses the same demand graph always
        else:
            add_demand_pairs(G, self.config.n_edges_demand, self.config.demand_capacity, self.config, self.random_gen_fix_max_flow)

        pg.plot(G, self.config.graph_path, distribution, self.config.destruction_precision, dim_ratio,
                self.config.destruction_show_plot, self.config.destruction_save_plot, self.config.seed, "TRU", co.PlotType.TRU, self.config.destruction_quantity)

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
        total_flow = sum([el[2] for el in gru.get_demand_edges(G)])
        packet_monitor = 0
        monitors_stats = set()
        demands_sat = {d: [] for d in get_demand_edges(G, is_capacity=False)}  # d1: [0, 1, 1, 0, 10] // demands_sat[d].append(0)

        # set as monitors all the nodes that are demand endpoints
        monitors_map = defaultdict(set)
        monitors_connections = defaultdict(set)
        monitors_non_connections = defaultdict(set)

        last_repaired_demand = None

        # ADD preliminary monitors

        for n1, n2, _ in get_demand_edges(G):
            G.nodes[n1][co.ElemAttr.IS_MONITOR.value] = True
            G.nodes[n2][co.ElemAttr.IS_MONITOR.value] = True
            monitors_stats |= {n1, n2}

            # does not look defined for only monitors
            monitors_map[n1] |= {(n1, n2)}
            monitors_map[n2] |= {(n1, n2)}

        self.config.monitors_budget_residual -= len(monitors_stats)
        fixed_paths = []
        pruning_paths_dict = {}
        # NEW STUFF
        mission_time = 0
        while mission_time < self.MISSION_DURATION:
            N_REPAIRS_ITERATION = []
            # check if the graph is still routbale on tot graph,
            if not is_feasible(G, is_fake_fixed=True):
                print("This instance is no more routable!")
                return rlog

            print("\n\n", "#" * 40, "BEGIN ITERATION {}".format(mission_time), "#" * 40)

            monitors, monitors_repaired, candidate_monitors_dem = mon.new_monitoring_add(G, self.config)
            monitors_map = mon.merge_monitor_maps(monitors_map, candidate_monitors_dem)  # F(n) -> [(d1, d2)]


            # >>>> PRUNING HERE
            monitoring = pruning_monitoring_dynamic(G,
                                                    self.config.monitoring_messages_budget,
                                                    monitors_map,
                                                    monitors_connections,
                                                    monitors_non_connections,
                                                    last_repaired_demand,
                                                    self.config)

            if monitoring is None:
                return rlog

            stats_packet_monitoring, demand_edges_to_repair, demand_edges_routed_flow, monitoring_paths, demand_edges_routed_flow_pp, pruned_paths = monitoring
            tomography_over_paths(G, elements_val_id, elements_id_val, self.config.UNK_prior, monitoring_paths)
            demand_edges = get_demand_edges(G, is_check_unsatisfied=True, is_residual=True)

            for p in pruned_paths:
                # updates the flow
                d_edge = gru.make_existing_edge(p[0][0], p[0][-1])
                path = p[0]
                pruning_paths_dict[d_edge] = path  # removed when destroy or remove

            print("PRUNING PATH", pruning_paths_dict)


            # -------------- 2. Decision recovery --------------
            if len(demand_edges) > 0:
                paths_proposed = frp.find_paths_to_repair(self.config.repairing_mode, G, demand_edges_to_repair,
                                                          get_supply_max_capacity(self.config),
                                                          is_oracle=self.config.is_oracle_baseline)

                path_to_fix = frpp.find_path_picker(self.config.picking_mode, G, paths_proposed,
                                                    self.config.repairing_mode, self.config,
                                                    is_oracle=self.config.is_oracle_baseline)

                # assert path_to_fix is not None
                self.do_increase_resistance(G, path_to_fix, self.config)

                d1, d2 = last_repaired_demand = make_existing_edge(path_to_fix[0], path_to_fix[-1])
                self.update_monitor_maps(d1, d2, monitors_non_connections, monitors_connections)

                fixed_nodes, fixed_edges = do_fix_path(G, path_to_fix)
                N_REPAIRS_ITERATION += fixed_nodes + fixed_edges

            mission_time = self.handle(G, N_REPAIRS_ITERATION, mission_time, rlog, pruning_paths_dict)
            print(mission_time, N_REPAIRS_ITERATION, rlog.show_log(mission_time), sep="\n\n")
            # time.sleep(5)

        return rlog  # stats_list

    def handle_no_repairs(self, G, mission_time, rlog, pruning_paths_dict, repair):
        # if self.is_event_happening(mission_time):
        event_happened, _, _ = self.handle_events(G, mission_time, pruning_paths_dict)

        d_edges_info = get_demand_edges_info(G)
        maximum_flow = sum([to for e1, e2, to, res, ro in d_edges_info])
        rlog.add_maximum_flow(maximum_flow)

        total_flow = sum([ro for e1, e2, to, res, ro in d_edges_info])
        rlog.add_total_flow(total_flow)

        rlog.add_event(event_happened.value)
        rlog.add_repairs([repair])

        # for e1, e2, to, res, ro in d_edges_info:
        #     rlog.add_demand_flow((e1, e2), ro)

    def handle(self, G, repairs, mission_time, rlog, pruning_paths_dict):

        if len(repairs) > 0:
            for t in range(len(repairs)):
                # self.handle_no_repairs(G, mission_time, rlog, pruning_paths_dict, repair=repairs[t])
                event_happened, _, _ = self.handle_events(G, mission_time, pruning_paths_dict)

                if t == 0 and mission_time > 0:  # first step
                    maximum_flow = rlog.get_maximum_flow(mission_time - 1)
                    rlog.add_maximum_flow(maximum_flow)
                    total_flow = rlog.get_total_flow(mission_time - 1)
                    rlog.add_total_flow(total_flow)
                    rlog.add_event(co.DynamicEvent.NONE.value)
                    rlog.add_repairs([repairs[t]])

                elif t < len(repairs)-1 and mission_time > 0:
                    maximum_flow = rlog.get_maximum_flow(mission_time-1)
                    rlog.add_maximum_flow(maximum_flow)
                    total_flow = rlog.get_total_flow(mission_time-1)
                    rlog.add_total_flow(total_flow)
                    rlog.add_event(rlog.get_event(mission_time-1))
                    rlog.add_repairs([repairs[t]])

                else:  # last step
                    self.handle_no_repairs(G, mission_time, rlog, pruning_paths_dict, repairs[t])

                mission_time += 1
                rlog.step()
        else:
            self.handle_no_repairs(G, mission_time, rlog, pruning_paths_dict, repair=None)
            mission_time += 1
            rlog.step()


        return mission_time

    @staticmethod
    def update_monitor_maps(d1, d2, monitors_non_connections, monitors_connections):
        # monitors are not connected PHASE 0
        monitors_non_connections[d1] |= {d2}
        monitors_non_connections[d2] |= {d1}

        monitors_connections[d1] -= {d2}
        monitors_connections[d2] -= {d1}

    @staticmethod
    def cancel_demand_edge(G, path_to_fix):
        print("Path with capacity 0, happened", path_to_fix)
        dd1, dd2 = make_existing_edge(path_to_fix[0], path_to_fix[-1])
        G.edges[dd1, dd2, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value] = 0

    @staticmethod
    def dynamic_destruction(G):
        N_EPICENTERS = 3
        HOPS = 3

        destr_nodes, destr_edges = set(), set()
        for i in range(N_EPICENTERS):
            id_epi = np.random.randint(len(G.nodes), size=1)
            id_node_epi = id_epi[0]  # G.nodes[id_epi[0]]
            destr_nodes_temp, destr_edges_temp = k_hop_destruction(G, id_node_epi, HOPS)
            destr_nodes |= destr_nodes_temp
            destr_edges |= destr_edges_temp
        return destr_nodes, destr_edges
        # WHEN DESTROY RESET CAPACITIES Viviana: and stop routing flow of demands traversing the newly broken elements

    def poisson_process_dynamic_events(self, rate, num_arrivals, arrival_time, config, event_type):
        # Viviana: tutta questa funzione (requires math and random)
        CONST_SEED = event_type * self.seed_fix_max_flow  # * config.seed
        util.set_seed(CONST_SEED)

        poi_process = []
        co = 0
        for i in range(num_arrivals):
            # Get the next probability value from Uniform(0,1)
            p = random.random()

            # Plug it into the inverse of the CDF of Exponential(_lamnbda)
            inter_arrival_time = -math.log(1.0 - p) / rate

            # Add the inter-arrival time to the running sum
            arrival_time += inter_arrival_time

            # print it all out
            # print(str(p) + ',' + str(inter_arrival_time) + ',' + str(arrival_time))
            cat = np.ceil(arrival_time)

            if co == 0 or (co > 0 and poi_process[co-1] != cat):
                poi_process += [cat]
                co += 1

        util.set_seed(config.seed)
        return poi_process

    @staticmethod
    def do_increase_resistance(G, path, config):
        """ increase the resistance of all the components in a path """
        n0 = path[0]
        G.nodes[n0][co.ElemAttr.RESISTANCE_TO_DESTRUCTION.value] = config.uniform_resistance_destruction_reset
        for i in range(len(path)-1):
            n1, n2 = make_existing_edge(path[i], path[i + 1])
            G.nodes[n2][co.ElemAttr.RESISTANCE_TO_DESTRUCTION.value] = config.uniform_resistance_destruction_reset
            G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESISTANCE_TO_DESTRUCTION.value] = config.uniform_resistance_destruction_reset

    def do_revert_routed_path(self, G, destr_nodes, destr_edges, reason: co.DynamicEvent, pruning_paths_dict):
        """ for paths that are broken, restore capacities """  # [1, 2, (3)] 60,  [1, 2, (3), 4] 40
        # [([124, 125, 122, 221, 233, 349], 30)]

        demand_edges_remove = []
        for ns, nd in gru.get_demand_edges(G, is_capacity=False):
            is_path_to_revert = False

            if reason == co.DynamicEvent.DESTROY:
                path = pruning_paths_dict[(ns, nd)]
                for i in range(len(path)-1):
                    n1, n2 = gru.make_existing_edge(path[i], path[i+1])
                    is_path_to_revert = n1 in destr_nodes or n2 in destr_nodes or (n1, n2) in destr_edges

            elif reason == co.DynamicEvent.REMOVE_NODE:
                is_path_to_revert = destr_nodes[0] in [ns, nd]

            elif reason == co.DynamicEvent.REMOVE_EDGE:
                dem1, dem2 = destr_edges[0]
                is_path_to_revert = dem1 in [ns, nd] or dem2 in [ns, nd]

            if is_path_to_revert:
                demand_edges_remove.append((ns, nd))

        print("GONNA DELETE EDGES", demand_edges_remove)
        for e1, e2, in demand_edges_remove:
            path = pruning_paths_dict[(e1, e2)]
            self.__do_revert_flow_path(G, path)
            del pruning_paths_dict[(e1, e2)]  # CAREFUL

        return demand_edges_remove

    def is_event_happening(self, time):
        return time in (self.destroy_times + self.new_node_times + self.new_edge_times + self.remove_edge_times + self.remove_node_times)

    @staticmethod
    def __do_revert_flow_path(G, path):
        ns, nd = make_existing_edge(path[0], path[-1])
        dem_edge_capacity = G.edges[ns, nd, co.EdgeType.DEMAND.value][co.ElemAttr.CAPACITY.value]
        G.edges[ns, nd, co.EdgeType.DEMAND.value][co.ElemAttr.RESIDUAL_CAPACITY.value] = dem_edge_capacity  # restoring the pruned quantity

        for i in range(len(path) - 1):
            n1, n2 = make_existing_edge(path[i], path[i + 1])
            G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value] += dem_edge_capacity

    def handle_events(self, G, mission_time, pruning_paths_dict):
        event_happened = co.DynamicEvent.NONE
        demand_edges_remove = []
        demand_edges_add = []
        if mission_time in self.destroy_times:
            destr_nodes, destr_edges = self.dynamic_destruction(G)
            print("DOING DESTRUCTION!!!", destr_nodes, destr_edges)
            demand_edges_remove = self.do_revert_routed_path(G, destr_nodes, destr_edges, co.DynamicEvent.DESTROY, pruning_paths_dict)
            event_happened = co.DynamicEvent.DESTROY

        elif mission_time in self.new_node_times:
            print("EVENT: ADDING DEMAND NODE!")

            demand_nodes = set(co.FIXED_DEMAND_NODES)
            cand_nodes = list(gru.get_supply_nodes(G) - demand_nodes)
            ind_new_node = self.random_gen_fix_max_flow.randint(low=0, high=len(cand_nodes))
            new_demand_node = cand_nodes[ind_new_node]

            ind_exist_node = self.random_gen_fix_max_flow.randint(low=0, high=len(demand_nodes))
            exist_demand_node = list(demand_nodes)[ind_exist_node]
            gru.make_demand_edge(G, new_demand_node, exist_demand_node, self.config.demand_capacity)

            demand_edges_add = [(new_demand_node, exist_demand_node)]
            # adding demand edge in stats
            # k = list(rlog.flow_per_demand.keys())[0]  # un arco qualsiasi
            # len_v = len(rlog.flow_per_demand[k])      # lunghezza flusso
            # rlog.flow_per_demand[(new_demand_node, exist_demand_node)] = list(np.zeros(shape=len_v))

            # stats["demands_sat"][(new_demand_node, exist_demand_node)] = list(np.zeros(shape=len_v))
            print("Time", iter, "new node added!", new_demand_node, ". All demand nodes", gru.get_demand_nodes(G))
            # stats["event"] = co.DynamicEvent.ADD_NODE.value  # destruction
            event_happened = co.DynamicEvent.ADD_NODE

        elif mission_time in self.new_edge_times:
            print("EVENT: ADDING DEMAND EDGE!")

            demand_nodes = co.FIXED_DEMAND_NODES
            demand_edges = gru.get_demand_edges(G, is_capacity=False)
            exist_node1, exist_node2 = None, None
            while (exist_node1, exist_node2) in demand_edges or (exist_node1, exist_node2) == (None, None):
                ind1_exist_node, ind2_exist_node = self.random_gen_fix_max_flow.choice(len(demand_nodes), 2, replace=False)
                exist_node1, exist_node2 = demand_nodes[ind1_exist_node], demand_nodes[ind2_exist_node]
            exist_node1, exist_node2 = gru.make_existing_edge(exist_node1, exist_node2)

            gru.make_demand_edge(G, exist_node1, exist_node2, self.config.demand_capacity)
            # adding demand edge in stats
            # k = list(stats["demands_sat"].keys())[0]
            # len_v = len(stats["demands_sat"][k])
            # stats["demands_sat"][(exist_node1, exist_node2)] = list(np.zeros(shape=len_v))

            # k = list(rlog.flow_per_demand.keys())[0]  # un arco qualsiasi
            # len_v = len(rlog.flow_per_demand[k])      # lunghezza flusso
            # rlog.flow_per_demand[(exist_node1, exist_node2)] = list(np.zeros(shape=len_v))
            demand_edges_add = [(exist_node1, exist_node2)]

            print("Time", iter, "new edge added!", (exist_node1, exist_node2), ". All demand edges", gru.get_demand_edges(G, is_capacity=False))
            # stats["event"] = co.DynamicEvent.ADD_EDGE.value  # destruction
            event_happened = co.DynamicEvent.ADD_EDGE

        elif mission_time in self.remove_node_times and len(gru.get_demand_edges(G)) > 0:
            print("EVENT: REMOVING DEMAND NODE!")

            demand_nodes = list(gru.get_demand_nodes(G))
            ind_new_node = self.random_gen_fix_max_flow.randint(low=0, high=len(demand_nodes))
            exist_node = demand_nodes[ind_new_node]

            demand_edges_remove = self.do_revert_routed_path(G, [exist_node], [], co.DynamicEvent.REMOVE_NODE, pruning_paths_dict)

            list_edges_to_remove = [el for el in get_demand_edges(G, is_capacity=False) if exist_node in el]
            for n1, n2 in list_edges_to_remove:
                gru.remove_demand_edge(G, n1, n2)
                # print("URCA INNER EDGE T", mission_time, n1, n2, self.config.seed)

                # stats["demands_sat"][(n1, n2)].append(-self.config.demand_capacity)  # removes the flow
            print("Time", iter, "removed demand node!", exist_node, ". All demand nodes",
                  gru.get_demand_edges(G, is_capacity=False))
            # stats["event"] = co.DynamicEvent.REMOVE_NODE.value  # destruction
            # routed_flow = stats["flow"]
            event_happened = co.DynamicEvent.REMOVE_NODE

        elif mission_time in self.remove_edge_times and len(gru.get_demand_edges(G)) > 0:
            print("EVENT: REMOVING DEMAND EDGE!")
            demand_edges = list(gru.get_demand_edges(G, is_capacity=False))
            ind_new_edge = self.random_gen_fix_max_flow.randint(low=0, high=len(demand_edges))
            n1, n2 = demand_edges[ind_new_edge]
            demand_edges_remove = self.do_revert_routed_path(G, [], [(n1, n2)], co.DynamicEvent.REMOVE_EDGE, pruning_paths_dict)

            # print("URCA EDGE T", mission_time, n1, n2, self.config.seed)
            gru.remove_demand_edge(G, n1, n2)
            # stats["demands_sat"][(n1, n2)].append(-self.config.demand_capacity)  # removes the flow
            print("Time", iter, "removed demand edge!", (n1, n2), ". All demand edges", gru.get_demand_edges(G, is_capacity=False))
            event_happened = co.DynamicEvent.REMOVE_EDGE

        return event_happened, demand_edges_remove, demand_edges_add
