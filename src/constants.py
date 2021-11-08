
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

# TODO: (1) NON INCHIODARE IL ROUTING, (2) OTTIMO, (3) BEST EFFORT (+ ottimizzazione)
# TODO: confronto con ISP e SHP, Deep (check)
# TODO: CLIQUE


class ElemAttr(Enum):
    # edges
    CAPACITY = 'capacity'
    RESIDUAL_CAPACITY = 'residual_capacity'
    TYPE = 'type'       # demand/supply
    WEIGHT = 'weight'
    SAT_DEM = 'sat_dem'  # a map for every supply edge: demand edge, percentage of satisfiability
    SAT_SUP = 'sat_sup'  # a list of edges that satisfy the demand

    # nodes
    LONGITUDE = 'Longitude'
    LATITUDE = 'Latitude'

    # all
    STATE_TRUTH = 'state'                  # BROKEN, WORKING  (INVISIBILE)
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


# constants
path_to_graph = "data/graphs/"

repair_cost = 1
epsilon = 10**-10


