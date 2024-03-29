
import src.constants as co

comparison_dims = {co.IndependentVariable.GRAPH: [
                        co.GraphName.MINNESOTA
                        ],
                   co.IndependentVariable.SEED: range(5),
                   co.IndependentVariable.ALGORITHM: [
                        co.Algorithm.PROTON,
                        co.Algorithm.PROTON_ORACLE,
                        co.Algorithm.CEDAR,
                        co.Algorithm.ST_PATH,
                        co.Algorithm.SHP,
                        co.Algorithm.ISR_SP,
                        co.Algorithm.ISR_MULTICOM
                        ]
                   }

indv_vary = {
    co.IndependentVariable.PROB_BROKEN: [.3, .4, .5, .6, .7, .8],
    co.IndependentVariable.MONITOR_BUDGET: [20, 22, 24, 26, 28, 30],
    co.IndependentVariable.N_DEMAND_EDGES: [4, 5, 6, 7, 8],
    co.IndependentVariable.FLOW_DEMAND: [10, 15, 20, 25, 30],
}

indv_fixed = {
    co.IndependentVariable.PROB_BROKEN: .8,
    co.IndependentVariable.MONITOR_BUDGET: 20,
    co.IndependentVariable.N_DEMAND_EDGES: 8,
    co.IndependentVariable.FLOW_DEMAND: 30,
}


