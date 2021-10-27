
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
    BROKEN = "red"
    WORKING = "green"
    UNK = "gray"
    NA = "#00ffff"  # used for demand edges, since Edge is a Node GREEN


class ElemAttr(Enum):
    # edges
    CAPACITY = 'capacity'
    RESIDUAL_CAPACITY = 'residual_capacity'

    # nodes
    LONGITUDE = 'Longitude'
    LATITUDE = 'Latitude'

    TYPE = 'type'       # demand/supply
    STATE = 'state'     # BROKEN, WORKING
    PRIOR_BROKEN = 'prior_broken'          # prior that component is broken
    POSTERIOR_BROKEN = 'posterior_broken'  # posterior that component is broken
    ID = 'id'

# constants
path_to_graph = "data/graphs/"
