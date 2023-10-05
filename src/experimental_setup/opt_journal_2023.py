
import src.constants as co

comparison_dims = {co.IndependentVariable.GRAPH: [
                        # co.GraphName.BELL_CANADA,
                        co.GraphName.PALMETTO
                        ],
                   co.IndependentVariable.ALGORITHM: [
                        co.Algorithm.MIN_TDS,
                        co.Algorithm.PROTON,
                        co.Algorithm.PROTON_ORACLE
                        ],
                   co.IndependentVariable.SEED: list(range(500, 600)) + [256, 257, 258, 266, 269, 273, 278, 202, 203, 206, 209, 210, 213, 214, 216, 219, 221, 232, 235, 237, 243, 245, 246, 248, 251, 253, 254],
}

indv_vary = {
    co.IndependentVariable.PROB_BROKEN: [.4, .45, .5, .55, .6],
    # co.IndependentVariable.MONITOR_BUDGET: [20, 22, 24, 26, 28, 30],
    # co.IndependentVariable.N_DEMAND_EDGES: [4, 5, 6, 7, 8],
    # co.IndependentVariable.FLOW_DEMAND: [10, 15, 20, 25, 30],
}

indv_fixed = {
    co.IndependentVariable.PROB_BROKEN: .8,
    co.IndependentVariable.MONITOR_BUDGET: 6,
    co.IndependentVariable.N_DEMAND_EDGES: 3,
    co.IndependentVariable.FLOW_DEMAND: 30,
}

# python -m src.main -set "setup_sept_23" -par 1
