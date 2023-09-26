# def debug_proton_opt_flow():
#     config = configuration.Configuration()
#     source = co.PATH_EXPERIMENTS
#     BENCHMARKS = [co.Algorithm.MIN_TDS, co.Algorithm.PROTON_ORACLE]
#     algo_names = [al.value.file_name for al in BENCHMARKS]
#     algo_names_plot = [al.value.plot_name for al in BENCHMARKS]
#     seeds = setup.comparison_dims[co.IndependentVariable.SEED]
#     OUTLIERS = 0
#     index_fixed_X = 1  # TODO: make it more generic
#
#     config.graph_dataset = co.GraphName.BELL_CANADA
#     x_var_k = co.IndependentVariable.PROB_BROKEN
#     X_var = [.4, .45, .5, .55, .6]  # independent variable [.1, .2, .3]
#     fixed_x = X_var[index_fixed_X]
#     print("Now varying", x_var_k, "as", X_var)
#
#     config.experiment_ind_var = x_var_k  # e.g. co.IndependentVariable.PROB_BROKEN
#     config.n_nodes_demand_clique = len(co.FIXED_DEMAND_NODES)
#
#     plot_name_ele = {co.IndependentVariable.PROB_BROKEN.name: "*",
#                      co.IndependentVariable.N_DEMAND_EDGES.name: "*",
#                      co.IndependentVariable.FLOW_DEMAND.name: "*",
#                      co.IndependentVariable.MONITOR_BUDGET.name: "*"}
#
#     config.n_edges_demand = setup.indv_fixed[co.IndependentVariable.N_DEMAND_EDGES]
#     plot_name_ele[co.IndependentVariable.N_DEMAND_EDGES.name] = config.n_edges_demand
#
#     config.demand_capacity = setup.indv_fixed[co.IndependentVariable.FLOW_DEMAND]
#     plot_name_ele[co.IndependentVariable.FLOW_DEMAND.name] = config.demand_capacity
#
#     config.monitors_budget = setup.indv_fixed[co.IndependentVariable.MONITOR_BUDGET]
#     plot_name_ele[co.IndependentVariable.MONITOR_BUDGET.name] = config.monitors_budget
#
#     path_prefix = source + "{}"  # "data/experiments/{}"
#     good_seeds = {256} #check_good_seeds(X_var, BENCHMARKS, seeds, 0, path_prefix, 500, config, False)
#     # 256, 257, 258, 266, 269, 273, 278, 202, 203, 206, 209, 210, 213, 214, 216, 219, 221, 232, 235, 237, 243, 245, 246, 248, 251, 253, 254
#     with PdfPages('debug_canada.pdf'.format(config.graph_dataset.name)) as pdf:
#         ndmp = X_var if x_var_k == co.IndependentVariable.N_DEMAND_EDGES else [config.n_edges_demand]
#         plot_Xvar_Ydems2(source, config, good_seeds, X_var, BENCHMARKS, variable_name=x_var_k, n_dem_edges=ndmp, plot_type=0,
#                          algo_names=algo_names, algo_names_plot=algo_names_plot, out_fig=pdf, title="Hi")
