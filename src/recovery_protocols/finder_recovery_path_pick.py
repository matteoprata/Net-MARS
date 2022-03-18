
from src import constants as co
from src.preprocessing import graph_utils as gu
import numpy as np
import random


def find_path_picker(id, G, paths, repair_mode, is_oracle=False):
    if id == co.ProtocolPickingPath.RANDOM:
        return __pick_random_repair_path(G, paths)

    elif id == co.ProtocolPickingPath.MAX_BOT_CAP:
        return __pick_cedar_repair_path(G, paths)

    elif id == co.ProtocolPickingPath.MIN_COST_BOT_CAP:
        return __pick_tomocedar_repair_path(G, paths, repair_mode, is_oracle=is_oracle)


def __pick_cedar_repair_path(G, paths):
    if len(paths) > 0:
        # PICK MAX CAPACITY
        # Map the path to its bottleneck capacity
        paths_caps = []
        for path_nodes in paths:
            cap = gu.get_path_residual_capacity(G, path_nodes)
            paths_caps.append(cap)

        path_id_to_fix = np.argmax(paths_caps)
        print("> Selected path to recover has capacity", paths_caps[path_id_to_fix])

        # 5. Repair edges and nodes
        path_to_fix = paths[path_id_to_fix]  # 1, 2, 3
        print("> Repairing path", path_to_fix)
        return path_to_fix


def __pick_tomocedar_repair_path(G, paths, repair_mode, is_oracle=False):
    if len(paths) > 0:
        # 3. Map the path to its bottleneck capacity
        paths_exp_cost = []

        is_oracle = is_oracle and repair_mode == co.ProtocolRepairingPath.MIN_COST_BOT_CAP

        for path_nodes in paths:  # TODO: randomize
            # min_cap = get_path_cost(G, path_nodes)
            exp_cost = gu.get_path_cost_VN(G, path_nodes, is_oracle)  # MINIMIZE expected cost of repair
            paths_exp_cost.append(exp_cost)

        # 4. Get the path that maximizes the minimum bottleneck capacity
        path_id_to_fix = np.argmin(paths_exp_cost)
        print("> Selected path to recover has capacity", gu.get_path_residual_capacity(G, paths[path_id_to_fix]))

        # 5. Repair edges and nodes
        path_to_fix = paths[path_id_to_fix]  # 1, 2, 3
        print("> Repairing path", path_to_fix)
        return path_to_fix


def __pick_random_repair_path(G, paths):
    if len(paths) > 0:
        # PICK RANDOM PATH
        path_id_to_fix = random.randint(0, len(paths) - 1)
        print("> Selected path to recover has capacity", gu.get_path_residual_capacity(G, paths[path_id_to_fix]))

        # 5. Repair edges and nodes
        path_to_fix = paths[path_id_to_fix]  # 1, 2, 3
        print("> Repairing path", path_to_fix)
        return path_to_fix
