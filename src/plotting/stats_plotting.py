import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from collections import defaultdict
# repairs iter  flow_cum  flow_perc  n_repairs  n_monitors  n_monitor_msg
import random
from src.utilities.util import min_max_normalizer
from src import main as ma
import src.constants as co


def plot_monitors_stuff(source, config, seeds_values, X_vals, algos, typep, x_position, algo_names):
    """

    :param source:
    :param config:
    :param seeds_values:
    :param X_vals:
    :param algos:
    :param typep:
    :return:
    """
    plot_name = {"n_monitors": "Number Monitors",
                 "n_monitor_msg": "Number Monitoring Messages",
                 "n_repairs": "Number Repairs"}

    path_prefix = source + "{}"

    Xlabels = {0: "Probability Broken", 1: "Number Demand Pairs", 2: "Demand Flow"}

    slices = []
    for al in algos:
        seeds_pbro = []
        algo, rep = al
        for pbro in X_vals:
            dfs = []  # many dfs as the seeds (columns to average on axis 0)
            for ss in seeds_values:
                if x_position == 0:
                    # varying probs
                    regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-bud|{}-pbro|{}-rep|{}-pik|{}-mop|{}.csv".format(
                        ss,
                        config.graph_dataset.name,
                        config.n_demand_pairs,
                        int(config.demand_capacity),
                        config.supply_capacity,
                        algo,
                        config.monitors_budget,
                        pbro,
                        rep[0], rep[1], rep[2])

                elif x_position == 1:
                    regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-bud|{}-pbro|{}-rep|{}-pik|{}-mop|{}.csv".format(
                        ss,
                        config.graph_dataset.name,
                        pbro,
                        int(config.demand_capacity),
                        config.supply_capacity,
                        algo,
                        config.monitors_budget,
                        config.destruction_quantity,
                        rep[0], rep[1], rep[2])

                elif x_position == 2:
                    # varying flow pp
                    regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-bud|{}-pbro|{}-rep|{}-pik|{}-mop|{}.csv".format(
                        ss,
                        config.graph_dataset.name,
                        config.n_demand_pairs,
                        int(pbro),
                        config.supply_capacity,
                        algo,
                        config.monitors_budget,
                        config.destruction_quantity,
                        rep[0], rep[1], rep[2])

                path = path_prefix.format(regex_fname)

                df = pd.read_csv(path)
                df = df[typep]
                df = pd.DataFrame([df.iloc[0]])
                dfs.append(df)

            seed_df = pd.concat(dfs, axis=1).T
            seeds_pbro.append(seed_df)
        slices.append(seeds_pbro)

    plt.figure(figsize=(10, 8))

    for i, sli in enumerate(slices):
        out = pd.concat(sli, axis=1).mean(axis=0)
        # out.columns = pbro_values
        # out = out
        # print(out)
        # exit()
        plt.plot(X_vals, out, label=algo_names[tuple(algos[i][1])])

    plt.legend()
    plt.xlabel(Xlabels[x_position])
    plt.ylabel(plot_name[typep])
    plt.grid(alpha=.4)
    plt.xticks(X_vals, X_vals)
    if typep == "n_monitor_msg":
        plt.yscale('log')

    # print(out)
    # plt.savefig(plot_name[typep] + str(time.time()) + ".png")
    plt.show()


def sample_file(seed, graph, np, dc, spc, alg, bud, pbro, rep, pik, mop):
    return "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-bud|{}-pbro|{}-rep|{}-pik|{}-mop|{}.csv".format(seed, graph, np, dc, spc, alg, bud, pbro, rep, pik, mop)


def plot_integral(source, config, seeds_values, X_var, algos, plot_type, x_position, outliers=0, algo_names=None):
    """
    :param source:
    :param config:
    :param seeds_values:
    :param X_var: is the "probability of destruction" or "number of nodes"
    :param algos:
    :param is_total:
    :param plot_type: 0 is cumulative flow , 1 total flow, 2 is targeted flow
    :return:
    """
    path_prefix = source + "{}"  # "data/experiments/{}"
    Xlabels = {0: "Probability Broken",
               1: "Number Demand Pairs",
               2: "Demand Flow"}

    MAX_STEPS = 150
    datas = np.empty(shape=(MAX_STEPS, len(seeds_values), len(algos), len(X_var)))
    datas[:] = np.nan

    # datas fill
    for j, x in enumerate(X_var):
        for i, algo in enumerate(algos):
            algo, rep = algo
            for k, ss in enumerate(seeds_values):
                # varying probability
                if x_position == 0:
                    MAX_TOTAL_FLOW = config.n_demand_pairs * config.demand_capacity
                    MAX_FLOW_STEPS = config.n_demand_pairs * config.demand_capacity * MAX_STEPS
                    regex_fname = sample_file(ss, config.graph_dataset.name, config.n_demand_pairs, int(config.demand_capacity),
                                              config.supply_capacity, algo, config.monitors_budget, x, rep[0], rep[1], rep[2])
                elif x_position == 1:
                    MAX_TOTAL_FLOW = int(x * config.demand_capacity)
                    MAX_FLOW_STEPS = x * config.demand_capacity * MAX_STEPS
                    regex_fname = sample_file(ss, config.graph_dataset.name, x, int(config.demand_capacity), config.supply_capacity,
                                              algo, config.monitors_budget, config.destruction_quantity, rep[0], rep[1], rep[2])

                elif x_position == 2:  # vary fpp
                    MAX_TOTAL_FLOW = int(config.n_demand_pairs * x)
                    MAX_FLOW_STEPS = config.n_demand_pairs * x * MAX_STEPS
                    regex_fname = sample_file(ss, config.graph_dataset.name, config.n_demand_pairs, int(x), config.supply_capacity,
                                              algo, config.monitors_budget, config.destruction_quantity, rep[0], rep[1], rep[2])


                path = path_prefix.format(regex_fname)
                df = pd.read_csv(path)["flow_cum"]
                df_len = df.shape[0]
                assert(df_len <= MAX_STEPS)
                datas[:df_len, k, i, j] = df.values

    # integral extension
    for j, x in enumerate(X_var):
        for i, algo in enumerate(algos):
            for k, ss in enumerate(seeds_values):
                vec = datas[:, k, i, j]
                mask = np.isnan(vec)
                max_val = np.nanmax(vec)
                datas[:, k, i, j] = np.where(~mask, vec, max_val)

    sum_flows = (datas.sum(axis=0) / MAX_FLOW_STEPS)
    avg_sum_flows = sum_flows.mean(axis=0)
    std_sum_flows = sum_flows.std(axis=0)

    # datas = np.empty(shape=(MAX_STEPS, len(seeds_values), len(algos), len(X_var)))
    # print(sum_flows[:,3,:])
    # print(avg_sum_flows[3,:])
    # print(std_sum_flows[3,:])
    #
    # print(algos)
    # exit()
    # discard outliers
    av_p_std = avg_sum_flows + std_sum_flows
    av_m_std = avg_sum_flows - std_sum_flows

    av_p_std = np.tile(av_p_std, (len(seeds_values), 1, 1))
    av_m_std = np.tile(av_m_std, (len(seeds_values), 1, 1))

    A = np.asarray((sum_flows <= av_p_std), dtype=int)
    B = np.asarray((sum_flows >= av_m_std), dtype=int)

    THRESHOLD = int(len(algos) * len(X_var) * outliers)

    # is how many times the seed was in the band
    normal_seeds = (A*B).sum(axis=(1, 2)) >= THRESHOLD   # a vector as big as the number of seeds, the sum says how many times the samples fall in the stds

    sum_flows = sum_flows[normal_seeds, :, :]
    avg_sum_flows = sum_flows.mean(axis=0)
    std_sum_flows = sum_flows.std(axis=0)

    if outliers > 0:
        print("discarded some seeds, now they are", sum_flows.shape[0])

    avg_max_flows = (datas.max(axis=0) / MAX_TOTAL_FLOW).mean(axis=0)
    std_max_flows = (datas.max(axis=0) / MAX_TOTAL_FLOW).std(axis=0)

    y_plot = avg_max_flows if plot_type == 1 else avg_sum_flows
    std_y_plot = std_max_flows if plot_type == 1 else std_sum_flows

    plt.figure(figsize=(10, 8))

    IS_TRUNCATE = True
    if IS_TRUNCATE:  # remove the last lines when all the algos reached the max
        mask_reach_max = (datas.mean(axis=1)[:, :, -1] == MAX_TOTAL_FLOW).sum(axis=1) == len(algos)
        datas = datas[~mask_reach_max, :]

    if plot_type == 2:
        avg_flow = datas.mean(axis=1)[:, :, -1] / MAX_TOTAL_FLOW  # last element
        for i, _ in enumerate(algos):
            plt.plot(np.arange(avg_flow.shape[0]), avg_flow[:, i], label=algo_names[tuple(algos[i][1])])
        plt.ylabel("Flow")
        plt.xlabel("Repair Steps")

    elif plot_type == 3:
        A1, A2 = 3, 0
        A1_avg_flow = datas.mean(axis=1)[:, A1, -1] / MAX_TOTAL_FLOW
        A2_avg_flow = datas.mean(axis=1)[:, A2, -1] / MAX_TOTAL_FLOW
        out = A1_avg_flow - A2_avg_flow
        label_out = "{} - {}".format(algo_names[tuple(algos[A1][1])], algo_names[tuple(algos[A2][1])])
        plt.plot(np.arange(out.shape[0]), out, label=label_out)
        plt.ylabel("Flow Difference")
        plt.xlabel("Repair Steps")

    else:
        for i, _ in enumerate(algos):
            plt.plot(X_var, y_plot[i], label=algo_names[tuple(algos[i][1])])
            # plt.fill_between(X_var, y_plot[i] - std_y_plot[i], y_plot[i] + std_y_plot[i], alpha=0.2)
        plt.ylabel("Total Flow" if plot_type else "Cumulative Flow")
        plt.xlabel(Xlabels[x_position])

    plt.legend()
    plt.grid(alpha=.4)
    # plt.savefig("flow" + str(time.time()) + ".png")
    plt.show()


def plot_Xflow_Yrepair(source, config, seeds_values, X_var, algos, x_position, fixed_percentage, algo_names):
    """
    :param source:
    :param config:
    :param seeds_values:
    :param X_var: is the "probability of destruction" or "number of nodes"
    :param algos:
    :param is_total:
    :return:
    """
    path_prefix = source + "{}"  # "data/experiments/{}"
    Xlabels = {0: "Probability Broken", 1: "Number Demand Nodes", 2: "Demand Flow"}

    MAX_STEPS = 150
    NORM_MAX_FLOW_STEPS = 100

    data_repairs = np.empty(shape=(NORM_MAX_FLOW_STEPS, len(seeds_values), len(algos), len(X_var)))

    # datas fill
    for j, x in enumerate(X_var):
        for i, algo in enumerate(algos):
            algo, rep = algo
            for k, ss in enumerate(seeds_values):
                # varying probability
                if x_position == 0:
                    MAX_TOTAL_FLOW = int(config.n_demand_pairs * config.demand_capacity)
                    regex_fname = sample_file(ss, config.graph_dataset.name, config.n_demand_pairs, int(config.demand_capacity),
                                              config.supply_capacity, algo, config.monitors_budget, x, rep[0], rep[1], rep[2])
                elif x_position == 1:
                    MAX_TOTAL_FLOW = int(x * config.demand_capacity)
                    regex_fname = sample_file(ss, config.graph_dataset.name, x, int(config.demand_capacity), config.supply_capacity,
                                              algo, config.monitors_budget, config.destruction_quantity, rep[0], rep[1], rep[2])

                elif x_position == 2:
                    MAX_TOTAL_FLOW = int(config.n_demand_pairs * x)
                    regex_fname = sample_file(ss, config.graph_dataset.name, config.n_demand_pairs, int(x), config.supply_capacity,
                                              algo, config.monitors_budget, config.destruction_quantity, rep[0], rep[1], rep[2])

                path = path_prefix.format(regex_fname)
                df = pd.read_csv(path)
                df_len = df.shape[0]
                assert(df_len <= MAX_STEPS)
                # last value must be NONE EDIT: no, if no repair is needed
                count_rep = df.groupby("flow_cum").size()

                pdf = pd.DataFrame()

                pdf["flow"] = [int(min_max_normalizer(int(v), 0, MAX_TOTAL_FLOW, 0, NORM_MAX_FLOW_STEPS)) for v in count_rep.index]  # scaling
                pdf["rep_count"] = np.cumsum(count_rep.values)
                pdf.iloc[1:, 1] = pdf.iloc[:-1, 1]
                pdf = pdf.drop([0])

                outr = np.empty(shape=(int(NORM_MAX_FLOW_STEPS)))
                outr[:] = np.nan
                outr[0] = 0
                for en, v in enumerate(pdf["flow"].values):
                    outr[v-1] = pdf["rep_count"].values[en]

                mask = np.isnan(outr)
                idx = np.where(~mask, np.arange(mask.shape[0]), 0)
                np.maximum.accumulate(idx, axis=0, out=idx)
                outr[mask] = outr[idx[mask]]
                # print(outr)
                # outr = [0 0 0 0 0 0 12 12 12 12 50 50 50 50 50 50]
                data_repairs[:, k, i, j] = outr

    avg_sum_flows = data_repairs.mean(axis=1)

    plt.figure(figsize=(10, 8))

    if fixed_percentage:
        avg_sum_flows = avg_sum_flows[:, :, fixed_percentage]  # fix a destruction
        for i, _ in enumerate(algos):
            plt.plot(np.arange(NORM_MAX_FLOW_STEPS)/100, avg_sum_flows[:, i], label=algo_names[tuple(algos[i][1])])
        plt.xlabel("Normalized flow ({})".format(MAX_TOTAL_FLOW))
        plt.ylabel("Number of repairs")
    else:
        avg_sum_flows = avg_sum_flows.sum(axis=0)
        for i, _ in enumerate(algos):
            plt.plot(X_var, avg_sum_flows[i], label=algo_names[tuple(algos[i][1])])
        plt.xlabel(Xlabels[x_position])
        plt.ylabel("Cumulative Repairs")

    plt.legend()
    plt.grid(alpha=.4)
    time.time()
    # plt.savefig("flow-repair" + str(time.time()) + ".png")
    plt.show()


def plotting_data():
    config = ma.setup_configuration()
    seeds = set(list(range(31, 40)))

    dis_uni = [.1, .2, .3, .4, .5, .6]
    n_pairs = [2, 3, 4, 5, 6]
    fpp = [2, 5, 10, 15, 20]  # 25 pp is not ok with 30 edge flow
    algos = []

    x_axis = {# co.IndependentVariable.PROB_BROKEN: dis_uni,
              # co.IndependentVariable.N_DEMAND_EDGES: n_pairs,
              co.IndependentVariable.FLOW_DEMAND: fpp
              }

    algo_names = {(0, 4, 5): "ST-PATH",
                  (1, 5, 5): "CEDAR-LIKE",
                  (2, 2, 4): "TOMO-CEDAR",
                  (2, 2, 3): "ORACLE"}

    algos += [("CEDARNEW", [i, j, k]) for i in [0] for j in [4] for k in [5]]
    algos += [("CEDARNEW", [i, j, k]) for i in [1] for j in [5] for k in [5]]
    algos += [("CEDARNEW", [i, j, k]) for i in [2] for j in [2] for k in [3]]
    algos += [("CEDARNEW", [i, j, k]) for i in [2] for j in [2] for k in [4]]

    source = "data/experiments/"
    OUTLIERS = 0

    for name, xa in x_axis.items():
        i, _ = name.value
        print("Now varying", i, "as", xa)

        if name == co.IndependentVariable.PROB_BROKEN:  # vary prob broken fix n_pairs, ffp
            config.n_demand_pairs = 5  # 5 is ok
            config.monitors_budget = 25
            config.supply_capacity = (30, None)

        elif name == co.IndependentVariable.N_DEMAND_EDGES:  # vary n_pairs fix prob and ffp
            config.destruction_quantity = 0.6  # 0.6 is ok
            config.monitors_budget = 25
            config.supply_capacity = (30, None)

        elif name == co.IndependentVariable.FLOW_DEMAND:  # vary fpp fix prob and n_pais
            config.destruction_quantity = 0.5
            config.n_demand_pairs = 5
            config.supply_capacity = (30, None)
            config.monitors_budget = 25

        plot_Xflow_Yrepair(source, config, seeds, xa, algos, x_position=i, fixed_percentage=-1, algo_names=algo_names)
        # plot_Xflow_Yrepair(source, config, seeds, xa, algos, x_position=i, fixed_percentage=None, algo_names=algo_names)

        plot_integral(source, config, seeds, xa, algos, plot_type=3, x_position=i, outliers=OUTLIERS, algo_names=algo_names)
        plot_integral(source, config, seeds, xa, algos, plot_type=2, x_position=i, outliers=OUTLIERS, algo_names=algo_names)
        plot_integral(source, config, seeds, xa, algos, plot_type=1, x_position=i, outliers=OUTLIERS, algo_names=algo_names)
        plot_integral(source, config, seeds, xa, algos, plot_type=0, x_position=i, outliers=OUTLIERS, algo_names=algo_names)

        plot_monitors_stuff(source, config, seeds, xa, algos, typep="n_monitor_msg", x_position=i, algo_names=algo_names)
        plot_monitors_stuff(source, config, seeds, xa, algos, typep="n_monitors", x_position=i, algo_names=algo_names)
        plot_monitors_stuff(source, config, seeds, xa, algos, typep="n_repairs", x_position=i, algo_names=algo_names)


if __name__ == '__main__':
    plotting_data()
