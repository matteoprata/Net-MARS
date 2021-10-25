
from enum import Enum


class GraphName(Enum):
    PALMETTO = "PALMETTO.gml"
    MINNESOTA = "MINNESOTA.gml"


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
    NA = "blue"  # used for demand edges, since Edge is a Node


class NodeLabels(Enum):
    STATE = "state"


class ElemAttr(Enum):
    CAPACITY = 'capacity'
    LONGITUDE = 'Longitude'
    LATITUDE = 'Latitude'

