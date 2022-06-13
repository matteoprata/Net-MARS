
import numpy as np
import random
import sys
import os
import json
import traceback
from src import constants as co
from src.preprocessing import graph_utils as gu
import json
import pickle


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


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


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)


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


def write_pickle(dictionary, fname):
    with open(fname, 'wb') as handle:
        pickle.dump(dictionary, handle)


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
