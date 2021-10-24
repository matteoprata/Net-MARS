
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal


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