
from enum import Enum


class GraphName(Enum):
    PALMETTO = "PALMETTO.gml"
    MINNESOTA = "MINNESOTA.gml"
    UNINET = "uninet.gml"
    BELL_SOUTH = "bell_south.gml"
    BELL_CANADA = "bell_canada.gml"
    SINET = "sinet.gml"
    INTEROUTE = "interoute.gml"
    NTT = "ntt.gml"
    DIAL = "dialtelecomcz.gml"
    COLT = "colt.gml"
    COGE = "cogentco.gml"


class PriorKnowledge(Enum):
    FULL = 0
    TOMOGRAPHY = 1


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
    WEIGHT_UNIT = 'weight_unit'

    SAT_DEM = 'sat_dem'  # a map for every supply edge: demand edge, percentage of satisfiability
    SAT_SUP = 'sat_sup'  # a list of edges that satisfy the demand

    # nodes
    LONGITUDE = 'Longitude'
    LATITUDE = 'Latitude'
    IS_MONITOR = 'is_monitor'
    CENTRALITY = 'centrality'

    # all
    STATE_TRUTH = 'state'                  # BROKEN0, WORKING1  (INVISIBILE)
    PRIOR_BROKEN = 'prior_broken'          # prior that component is broken
    POSTERIOR_BROKEN = 'posterior_broken'  # posterior that component is broken
    ID = 'id'


class GraphElement(Enum):
    EDGE = 0
    NODE = 1


class Knowledge(Enum):
    TRUTH = 0,
    KNOW = 1


class PlotType(Enum):
    TRU = 0
    KNO = 1
    ROU = 2


class AlgoName(Enum):
    CEDAR = "CEDAR"
    ISP = "ISP"
    SHP = "SHP"
    DEEP = "DEEP"
    TOMO_CEDAR = "TOMO_CEDAR"
    CEDARNEW = "CEDARNEW"


# constants
PATH_TO_GRAPH = "data/graphs/"

REPAIR_COST = 1
EPSILON = 10 ** -10

N_CORES = 8*2
