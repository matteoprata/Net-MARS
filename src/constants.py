from enum import Enum
import multiprocessing
import src.recovery_protocols.main_tomocedar_setup as main_tomocedar_setup
import src.recovery_protocols.main_ISR_setup as main_ISR_setup
# import src.recovery_protocols.main_cedar_setup as main_cedar_setup
import src.recovery_protocols.main_cedar_setup_FINE_LUGLIO as main_cedar_setup_FL
import src.recovery_protocols.main_reactive_tomocedar_setup as main_reactive_tomocedar_setup

import src.recovery_protocols.main_SHP_setup as main_SHP_setup

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
    NAME = "name"
    PLOT_MARKER = "marker"
    EXEC = "exec"


class Algorithm(Enum):
    TOMO_CEDAR = {AlgoAttributes.NAME: "TOMO_CEDAR",
                  AlgoAttributes.REPAIRING_PATH: ProtocolRepairingPath.MIN_COST_BOT_CAP,
                  AlgoAttributes.PICKING_PATH: ProtocolPickingPath.MIN_COST_BOT_CAP,
                  AlgoAttributes.MONITOR_PLACEMENT: ProtocolMonitorPlacement.BUDGET,
                  AlgoAttributes.MONITORING_TYPE: PriorKnowledge.TOMOGRAPHY,
                  AlgoAttributes.PLOT_MARKER: "D",
                  AlgoAttributes.EXEC: main_tomocedar_setup
                  }

    TOMO_CEDAR_DYN = {AlgoAttributes.NAME: "TOMO_CEDAR_DYN",
                      AlgoAttributes.REPAIRING_PATH: ProtocolRepairingPath.MIN_COST_BOT_CAP,
                      AlgoAttributes.PICKING_PATH: ProtocolPickingPath.MIN_COST_BOT_CAP,
                      AlgoAttributes.MONITOR_PLACEMENT: ProtocolMonitorPlacement.BUDGET,
                      AlgoAttributes.MONITORING_TYPE: PriorKnowledge.TOMOGRAPHY,
                      AlgoAttributes.PLOT_MARKER: "D",
                      AlgoAttributes.EXEC: main_reactive_tomocedar_setup
                      }

    ORACLE = {AlgoAttributes.NAME: "ORACLE",
              AlgoAttributes.REPAIRING_PATH: ProtocolRepairingPath.MIN_COST_BOT_CAP,
              AlgoAttributes.PICKING_PATH: ProtocolPickingPath.MIN_COST_BOT_CAP,
              AlgoAttributes.MONITOR_PLACEMENT: ProtocolMonitorPlacement.NONE,
              AlgoAttributes.MONITORING_TYPE: PriorKnowledge.TOMOGRAPHY,
              AlgoAttributes.PLOT_MARKER: "o",
              AlgoAttributes.EXEC: main_tomocedar_setup
              }

    ST_PATH = {AlgoAttributes.NAME: "ST_PATH",
               AlgoAttributes.REPAIRING_PATH: ProtocolRepairingPath.SHORTEST_MINUS,
               AlgoAttributes.PICKING_PATH: ProtocolPickingPath.RANDOM,
               AlgoAttributes.MONITOR_PLACEMENT: ProtocolMonitorPlacement.NONE,
               AlgoAttributes.MONITORING_TYPE: PriorKnowledge.DUNNY_IP,
               AlgoAttributes.PLOT_MARKER: "v",
               AlgoAttributes.EXEC: main_tomocedar_setup
               }

    _ignore_ = ['_dict']
    _dict = {AlgoAttributes.REPAIRING_PATH: None,
             AlgoAttributes.PICKING_PATH: None,
             AlgoAttributes.MONITORING_TYPE: None
             }

    #
    CEDAR = {**{AlgoAttributes.NAME: "CEDAR",
                AlgoAttributes.PLOT_MARKER: "s",
                AlgoAttributes.MONITOR_PLACEMENT: None,
                AlgoAttributes.EXEC: main_cedar_setup_FL  # main_cedar_setup
                }, **_dict}
    #
    SHP = {**{AlgoAttributes.NAME: "SHP",
              AlgoAttributes.PLOT_MARKER: "p",
              AlgoAttributes.MONITOR_PLACEMENT: None,
              AlgoAttributes.EXEC: main_SHP_setup}, **_dict}

    ISR_SP = {**{AlgoAttributes.NAME: "ISR_SP",
                 AlgoAttributes.PLOT_MARKER: ">",
                 AlgoAttributes.MONITOR_PLACEMENT: None,
                 AlgoAttributes.EXEC: main_ISR_setup}, **_dict}

    ISR_MULTICOM = {**{AlgoAttributes.NAME: "ISR_MULTICOM",
                       AlgoAttributes.PLOT_MARKER: "<",
                       AlgoAttributes.MONITOR_PLACEMENT: None,
                       AlgoAttributes.EXEC: main_ISR_setup}, **_dict}


class IndependentVariable(Enum):
    PROB_BROKEN = 0, "Percentage Broken Elements"
    N_DEMAND_EDGES = 1, "Number Demand Edges"
    FLOW_DEMAND = 2, "Flow Demand Pair"
    MONITOR_BUDGET = 3, "Flow Demand Pair"
    SEED = 4, "Seed"
    ALGORITHM = 5, "Algorithm"
    IND_VAR = 6, "Independent Variable"


# constants
PATH_TO_GRAPH = "data/graphs/"
PATH_TO_FAILED_TESTS = "data/failed_tests_{}.txt"
PATH_EXPERIMENTS = "data/experiments/"

REPAIR_COST = 500
REPAIR_INTERVENTION = 100

EPSILON = 10 ** -10

N_CORES = multiprocessing.cpu_count() - 1

MINNESOTA_STP_BACKBONE = [(78, 79), (125, 86), (86, 193), (193, 188), (188, 320), (186, 188), (186, 320), (79, 349),
                          (349, 125), (78, 72), (1, 320), (1, 319), (125, 652), (652, 559), (559, 564),  # VERT SX
                          (106, 312), (312, 373),              # VERT DX
                          (72, 564), (564, 373), (373, 1),     # UP HOR
                          (320, 106), (106, 652), (652, 349),  # MID-LOW HOR
                          (319, 312), (312, 559), (559, 349)   # MI-DUP HOR
                          ]
