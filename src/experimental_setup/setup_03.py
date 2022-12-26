
import src.constants as co

comparison_dims = {co.IndependentVariable.GRAPH: [
                        co.GraphName.MINNESOTA
                        ],
                   co.IndependentVariable.SEED: range(1),
                   co.IndependentVariable.ALGORITHM: [
                        co.Algorithm.PROTON
                        ]
                   }

indv_vary = {
    co.IndependentVariable.PROB_BROKEN: [.3],
}

indv_fixed = {
    co.IndependentVariable.PROB_BROKEN: .8,
    co.IndependentVariable.MONITOR_BUDGET: 20,
    co.IndependentVariable.N_DEMAND_EDGES: 5,
    co.IndependentVariable.FLOW_DEMAND: 30,
}
