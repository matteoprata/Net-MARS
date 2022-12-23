
from src import constants as co
from src.preprocessing import graph_utils as gu
from src.recovery_protocols import finder_recovery_path as re

import numpy as np
import random


def find_path_picker(id, G, paths, repair_mode, config, is_oracle=False):

    if len(paths) == 0:
        print("No paths to recover.")
        return

    if id == co.ProtocolPickingPath.RANDOM:
        return __pick_random_repair_path(G, paths)

    elif id == co.ProtocolPickingPath.MAX_BOT_CAP:
        return __pick_cedar_repair_path(G, paths)

    elif id == co.ProtocolPickingPath.MIN_COST_BOT_CAP:
        return __pick_tomocedar_repair_path(G, paths, repair_mode, config, is_oracle=is_oracle)

    # elif id == co.ProtocolPickingPath.MAX_INTERSECT:
    #     return __pick_max_intersection(G, paths, repair_mode, is_oracle)

    elif id == co.ProtocolPickingPath.SHORTEST:
        return __pick_shortest(paths)

    elif id == co.ProtocolPickingPath.CEDAR_LIKE_MIN:
        return __pick_cedar_like_min(G, paths)


def __pick_max_intersection(G, paths, repair_mode, is_oracle):
    """ Picks th epath that has more elements in common with the other paths. """

    path_elements = dict()  # path id: [n1, n2, n3, (n1,n3)]
    print(paths)
    for pid, pp in enumerate(paths):
        elements = set()
        for i in range(len(pp)-1):
            n1, n2 = gu.make_existing_edge(pp[i], pp[i + 1])
            elements.add(n1)
            elements.add(n2)
            elements.add((n1, n2))
        path_elements[pid] = elements

    # [n1, n2, n3], [n1, n2, n5], [n1, n2, n5, n6]
    # PATH[i] intersect (union path[j] for all j)
    commons_path_elements = {i: 0 for i in range(len(path_elements))}  # path id : his max intersection
    for i in path_elements:
        elements_p = set()
        for j in path_elements:
            if i != j:
                elements_p |= path_elements[j]
        commons_path_elements[i] = len(path_elements[i].intersection(elements_p))

    container_items = sorted(commons_path_elements.items(), key=lambda x: x[1], reverse=True)  # (p, len)

    # TIE BREAKING: repair the least expected cost
    # group dict by key
    n_intersections_paths = dict()  # k: number of intersections, v: paths
    for key, value in container_items:
        n_intersections_paths.setdefault(value, []).append(key)

    intersected_paths_ids = list(n_intersections_paths.items())[0][1]  # list of ids of the paths to possibly repair
    intersected_paths = [paths[i] for i in intersected_paths_ids]

    if len(intersected_paths_ids) > 1:
        path_to_fix = __pick_tomocedar_repair_path(G, intersected_paths, repair_mode, is_oracle)
    else:
        pid = intersected_paths_ids[0]
        path_to_fix = paths[pid]

    return path_to_fix


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


def __pick_tomocedar_repair_path(G, paths, repair_mode, config,  is_oracle=False):
    if len(paths) > 0:
        # 3. Map the path to its bottleneck capacity
        paths_exp_cost = []

        for path_nodes in paths:  # TODO: randomize
            # min_cap = get_path_cost(G, path_nodes)
            print(path_nodes)
            exp_cost = gu.get_path_cost_VN(G, path_nodes, is_oracle, config)  # MINIMIZE expected cost of repair
            paths_exp_cost.append(exp_cost)
            print("COST >", exp_cost)
            print()

        # 4. Get the path that maximizes the minimum bottleneck capacity
        path_id_to_fix = np.argmin(paths_exp_cost)
        print("> Selected path to recover has capacity", gu.get_path_residual_capacity(G, paths[path_id_to_fix]))

        # 5. Repair edges and nodes
        path_to_fix = paths[path_id_to_fix]  # 1, 2, 3
        print("> Repairing path", path_to_fix)
        print("cost >", paths_exp_cost[path_id_to_fix])
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


def __pick_shortest(paths):
    """ The path with few hops. """
    if len(paths) > 0:
        pp_id = np.argmin([len(p) for p in paths])
        return paths[pp_id]


def __pick_cedar_like_min(G, paths):
    if len(paths) > 0:
        # 3. Map the path to its bottleneck capacity
        paths_exp_cost = []

        for path_nodes in paths:  # TODO: randomize
            # min_cap = get_path_cost(G, path_nodes)
            exp_cost = gu.get_path_cost_cedarlike(G, path_nodes)  # MINIMIZE expected cost of repair
            paths_exp_cost.append(exp_cost)

        # 4. Get the path that maximizes the minimum bottleneck capacity
        path_id_to_fix = np.argmin(paths_exp_cost)
        print("> Selected path to recover has capacity", gu.get_path_residual_capacity(G, paths[path_id_to_fix]))

        # 5. Repair edges and nodes
        path_to_fix = paths[path_id_to_fix]  # 1, 2, 3
        print("> Repairing path", path_to_fix)
        print("cost >", paths_exp_cost[path_id_to_fix])
        return path_to_fix
