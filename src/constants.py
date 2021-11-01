
from enum import Enum


class GraphName(Enum):
    PALMETTO = "PALMETTO.gml"
    MINNESOTA = "MINNESOTA.gml"
    UNINET = "uninet.gml"
    BELL_SOUTH = "bell_south.gml"
    BELL_CANADA = "bell_canada.gml"
    SINET = "sinet.gml"


class Destruction(Enum):
    UNIFORM = "random"
    GAUSSIAN = "gaussian"
    COMPLETE = "complete"


class EdgeType(Enum):
    DEMAND = 1
    SUPPLY = 0


class NodeState(Enum):
    BROKEN = 1
    WORKING = 0
    UNK = .5
    NA = -1  # used for demand edges, since Edge is a Node GREEN


class ElemAttr(Enum):
    # edges
    CAPACITY = 'capacity'
    RESIDUAL_CAPACITY = 'residual_capacity'
    TYPE = 'type'       # demand/supply
    WEIGHT = 'weight'

    # nodes
    LONGITUDE = 'Longitude'
    LATITUDE = 'Latitude'

    STATE_TRUTH = 'state'                      # BROKEN, WORKING  (INVISIBILE)
    PRIOR_BROKEN = 'prior_broken'          # prior that component is broken
    POSTERIOR_BROKEN = 'posterior_broken'  # posterior that component is broken
    ID = 'id'


class GraphElement(Enum):
    EDGE = 0
    NODE = 1


class Knowledge(Enum):
    TRUTH = 0,
    KNOW = 1


# constants
path_to_graph = "data/graphs/"

repair_cost = 1
epsilon = 10**-10
