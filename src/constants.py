from enum import Enum
import multiprocessing
import matplotlib.pyplot as plt


GUROBI_STATUS = {1: 'LOADED', 2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED', 6: 'CUTOFF',
                 7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT', 9: 'TIME_LIMIT', 10: 'SOLUTION_LIMIT', 11: 'INTERRUPTED',
                 12: 'NUMERIC', 13: 'SUBOPTIMAL', 14: 'INPROGRESS', 15: 'USER_OBJ_LIMIT'}


def sample_color(index, cmap='tab10'):
    # 1. Choose your desired colormap
    cmap = plt.get_cmap(cmap)

    # 2. Segmenting the whole range (from 0 to 1) of the color map into multiple segments
    colors = [cmap(x) for x in range(cmap.N)]
    assert index < cmap.N

    # 3. Color the i-th line with the i-th color, i.e. slicedCM[i]
    color = colors[index]
    return color


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
    TOMOGRAPHY = 1
    DUNNY_IP = 2


class Destruction(Enum):
    UNIFORM = "random"
    GAUSSIAN = "gaussian"
    COMPLETE = "complete"
    GAUSSIAN_PROGRESSIVE = "gaussian_progressive"


class EdgeType(Enum):
    DEMAND = 1
    SUPPLY = 0


class NodeState(Enum):
    BROKEN = 1
    WORKING = 0
    UNK = None  # dyanamic
    NA = -1  # used for demand edges, since Edge is a Node GREEN


class ElemAttr(Enum):
    # edges
    CAPACITY = 'capacity'
    RESIDUAL_CAPACITY = 'residual_capacity'
    TYPE = 'type'  # demand/supply
    WEIGHT = 'weight'
    WEIGHT_UNIT = 'weight_unit'
    IS_BACKBONE = 'backbone'

    SAT_DEM = 'sat_dem'  # a map for every supply edge: demand edge, percentage of satisfiability
    SAT_SUP = 'sat_sup'  # a list of edges that satisfy the demand

    # nodes
    LONGITUDE = 'Longitude'
    LATITUDE = 'Latitude'
    IS_MONITOR = 'is_monitor'
    CENTRALITY = 'centrality'
    IS_EPICENTER = 'is_epicenter'

    # all
    STATE_TRUTH = 'state'  # ENUMERATOR  (INVISIBILE)
    PRIOR_BROKEN = 'prior_broken'  # prior that component is broken
    POSTERIOR_BROKEN = 'posterior_broken'  # posterior that component is broken
    ID = 'id'
    RESISTANCE_TO_DESTRUCTION = 'resistence'


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


class ProtocolRepairingPath(Enum):
    SHORTEST_PRO = 0
    MAX_BOT_CAP = 1
    MIN_COST_BOT_CAP = 2
    IP = 3
    AVERAGE = 4
    SHORTEST_MINUS = 5


class ProtocolPickingPath(Enum):
    RANDOM = 0
    MAX_BOT_CAP = 1
    MIN_COST_BOT_CAP = 2
    MAX_INTERSECT = 3

    SHORTEST = 4
    CEDAR_LIKE_MIN = 5


class ProtocolMonitorPlacement(Enum):
    STEP_BY_STEP = 0
    BUDGET_W_REPLACEMENT = 1
    STEP_BY_STEP_INFINITE = 6
    BUDGET = 4
    NONE = 5


class AlgoAttributes(Enum):
    REPAIRING_PATH = "ProtocolRepairingPath"
    PICKING_PATH = "ProtocolPickingPath"
    MONITOR_PLACEMENT = "ProtocolMonitorPlacement"
    MONITORING_TYPE = "PriorKnowledge"
    FILE_NAME = "name"
    PLOT_MARKER = "marker"
    EXEC = "exec"
    COLOR = "color"
    PLOT_NAME = "plot_name"
    LINE_STYLE = "lstyle"


class IndependentVariable(Enum):
    PROB_BROKEN = 0, "Percentage Broken Elements"
    N_DEMAND_EDGES = 1, "Number Demand Edges"
    FLOW_DEMAND = 2, "Flow Demand Pair"
    MONITOR_BUDGET = 3, "Monitor Total Budget"
    SEED = 4, "Seed"
    ALGORITHM = 5, "Algorithm"
    GRAPH = 7, "Graph"


# constants
PATH_TO_GRAPH = "data/graphs/"
PATH_TO_FAILED_TESTS = "data/failed_tests_{}.txt"
PATH_TO_FAILED_SEEDS = "data/failed_seeds.txt"
PATH_EXPERIMENTS = "data/experiments/"

REPAIR_COST = 500
REPAIR_INTERVENTION = 100

EPSILON = 10 ** -10

N_CORES = multiprocessing.cpu_count()

MINNESOTA_STP_BACKBONE = [(78, 79), (125, 86), (86, 193), (193, 188), (188, 320), (186, 188), (186, 320), (79, 349),
                          (349, 125), (78, 72), (1, 320), (1, 319), (125, 652), (652, 559), (559, 564),  # VERT SX
                          (106, 312), (312, 373),              # VERT DX
                          (72, 564), (564, 373), (373, 1),     # UP HOR
                          (320, 106), (106, 652), (652, 349),  # MID-LOW HOR
                          (319, 312), (312, 559), (559, 349)   # MI-DUP HOR
                          ]

# in minnesota they are the highest degree nodes, from which backbone may start
FIXED_DEMAND_NODES = [320, 78, 1, 349, 124, 564, 315, 186]


from src.recovery_protocols.protocols.PRoTOn import PRoTOn
from src.recovery_protocols.protocols.PRoTOnOracle import PRoTOnOracle
from src.recovery_protocols.protocols.CeDAR import CeDAR
from src.recovery_protocols.protocols.ShP import ShP
from src.recovery_protocols.protocols.RecShortestPath import RecShortestPath
from src.recovery_protocols.protocols.PRoTOnDyn import PRoTOnDyn
from src.recovery_protocols.protocols.ISRShortestPath import ISRShortestPath
from src.recovery_protocols.protocols.ISRMultiCommodity import ISRMultiCommodity


class Algorithm(Enum):
    PROTON = PRoTOn
    PROTON_ORACLE = PRoTOnOracle
    PROTON_DYN = PRoTOnDyn
    ST_PATH = RecShortestPath
    ISR_SP = ISRShortestPath
    ISR_MULTICOM = ISRMultiCommodity
    CEDAR = CeDAR
    SHP = ShP


from src.experimental_setup import setup_01, setup_02, setup_03


class Setups(Enum):
    SETUP_01 = setup_01
    SETUP_02 = setup_02
    SETUP_03 = setup_03
