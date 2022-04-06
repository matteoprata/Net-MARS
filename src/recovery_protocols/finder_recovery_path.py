
import src.utilities.util_routing_stpath as mxv
from src import constants as co
from src.preprocessing import graph_utils as gu


def find_paths_to_repair(id, G, demand_edges_to_repair, max_sup_cap, is_oracle=False):
    paths = []
    for n1, n2 in demand_edges_to_repair:
        SG = gu.get_supply_graph(G)

        if id == co.ProtocolRepairingPath.SHORTEST:
            path = mxv.protocol_stpath_capacity(SG, n1, n2)
        elif id == co.ProtocolRepairingPath.IP:
            path, _, _ = mxv.protocol_routing_IP(SG, n1, n2)
        elif id == co.ProtocolRepairingPath.MIN_COST_BOT_CAP:  # TOMO-CEDAR
            residual_demand = gu.get_residual_demand(G)
            path, _, _ = mxv.protocol_repair_min_exp_cost(SG, n1, n2, residual_demand, max_sup_cap, is_oracle=is_oracle)
        elif id == co.ProtocolRepairingPath.MAX_BOT_CAP:  # CEDAR
            path, _, _ = mxv.protocol_repair_cedarlike(SG, n1, n2)
        elif id == co.ProtocolRepairingPath.AVERAGE:
            path, _, _ = mxv.protocol_repair_AVG_COST(SG, n1, n2, is_oracle=is_oracle)
        else:
            path = None

        paths.append(path)
    return paths
