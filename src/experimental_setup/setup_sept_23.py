
import src.constants as co

comparison_dims = {co.IndependentVariable.GRAPH: [
                        co.GraphName.BELL_CANADA,
                        co.GraphName.PALMETTO
                        ],
                   co.IndependentVariable.ALGORITHM: [
                        co.Algorithm.MIN_TDS,
                        co.Algorithm.PROTON,
                        co.Algorithm.PROTON_ORACLE
                        ],
                   co.IndependentVariable.SEED: range(200, 280),
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
