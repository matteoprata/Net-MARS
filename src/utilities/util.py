
import numpy as np
import random
import sys
import os
import src.constants as co
from src.preprocessing import network_utils as gu
import json
import pickle
from multiprocessing import Pool
import signal
import matplotlib.pyplot as plt

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def sample_marker(index):
    MARKERS = ["p", "s", "P", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8"]
    return MARKERS[index]


def sample_pattern(index):
    MARKERS = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'] + ['/o', '\\|', '|*', '-\\', '+o', 'x*', 'o-', 'O|', 'O.', '*-']
    return MARKERS[index]


def sample_line(index):
    MARKERS = ['-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted', 'loosely dotted', 'densely dotted', 'loosely dashed', 'densely dashed', 'loosely dashdotted', 'densely dashdotted', 'loosely dashdotdotted', 'dashdotdotted', 'densely dashdotdotted']
    return MARKERS[index]


def sample_color(index, cmap='tab10'):
    # 1. Choose your desired colormap
    cmap = plt.get_cmap(cmap)

    # 2. Segmenting the whole range (from 0 to 1) of the color map into multiple segments
    colors = [cmap(x) for x in range(cmap.N)]
    assert index < cmap.N

    # 3. Color the i-th line with the i-th color, i.e. slicedCM[i]
    color = colors[index]
    return color


def execute_parallel_processes(func_exe, func_args: list, n_cores: int=1):
    """
    Runs processes in parallel. Given the function to run and its arguments.
    :param func_exe: function to run.
    :param func_args: arguments.
    """

    # initializer = signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ignore CTRL+C in the worker process.
    with Pool(processes=n_cores) as pool:
        try:
            pool.starmap(func_exe, func_args)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()

    print("COMPLETED SUCCESSFULLY")


def is_distance_tolerated(perc_broken_sofar, destruction_quantity, tolerance):
    return abs(perc_broken_sofar - destruction_quantity) < tolerance


def min_max_normalizer(value, startLB, startUB, endLB=0, endUB=1):
    # Figure out how 'wide' each range is
    value = np.asarray(value)
    if not (value <= startUB).all() and (value >= startLB).all():

        print("ERROR violated normalization bounds", value, startLB, startUB)
        exit()

    leftSpan = startUB - startLB
    rightSpan = endUB - endLB
    # Convert the left range into a 0-1 range (float)
    valueScaled = (value - startLB) / leftSpan
    new_value = ((valueScaled * rightSpan) + endLB)
    return new_value


def detuple_list(li):
    """ li: [(el1, el2, el3...), ] -> [el1, el2, el3..., ]"""
    nuli = set()
    for tuple in li:
        for el in tuple:
            nuli.add(el)
    return nuli


def safe_exec(func, pars):
    try:
        out = func(*pars)
        return out
    except:
        # traceback.print_exception(*sys.exc_info())
        return None


def disable_print():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enable_print():
    sys.stdout = sys.__stdout__


def read_json(fname):
    with open(fname) as json_file:
        data = json.load(json_file)
        return data


def write_json(dictionary, fname):
    with open(fname, "w") as json_file:
        json.dump(dictionary, json_file)


def read_pickle(fname):
    with open(fname, 'rb') as handle:
        data = pickle.load(handle)
        return data


def read_file(fname):
    with open(fname, 'r') as handle:
        lis = [l.strip() for l in handle.readlines()]
    return lis


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def write_pickle(dictionary, fname):
    with open(fname, 'wb') as handle:
        pickle.dump(dictionary, handle)


def write_file(text, fname, is_append=False):
    with open(fname, "a" if is_append else "w") as myfile:
        myfile.write(text)


def nearest_value_index(value, list_values:list):
    nval = min(list_values, key=lambda x: abs(x - value))
    nval_index = list_values.index(nval) - 1 if nval > value else list_values.index(nval)
    if nval_index < 0 or nval_index >= len(list_values):
        return None
    return list_values[nval_index]


def save_porting_dictionary(G, fname):
    """ Stores the graph characteristics. """
    demand_edges_flow = {str((n1, n2)): c for n1, n2, c in gu.get_demand_edges(G)}
    normal_edges_flow = {str((n1, n2)): G.edges[n1, n2, tip][co.ElemAttr.CAPACITY.value] for n1, n2, tip in G.edges if tip == co.EdgeType.SUPPLY.value}

    normal_edges_stat = {str((n1, n2)): G.edges[n1, n2, tip][co.ElemAttr.STATE_TRUTH.value] for n1, n2, tip in G.edges if tip == co.EdgeType.SUPPLY.value}
    normal_nodes_stat = {str(n): G.nodes[n][co.ElemAttr.STATE_TRUTH.value] for n in G.nodes}

    out = {"demand_edges_flow": demand_edges_flow,
           "normal_edges_flow": normal_edges_flow,
           "normal_edges_stat": normal_edges_stat,
           "normal_nodes_stat": normal_nodes_stat
           }

    with open(fname, 'w') as f:
        json.dump(out, f)



