
import numpy as np
import random
import sys
import os

import traceback
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
        print("ERROR", value, startLB, startUB)
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
        return True, out
    except:
        traceback.print_exception(*sys.exc_info())
        return False, None


def block_print():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enable_print():
    sys.stdout = sys.__stdout__