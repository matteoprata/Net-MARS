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
from itertools import combinations
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import sem

np.set_printoptions(suppress=True)


def plot_monitors_stuff(source, config, seeds_values, X_vals, algos, typep, x_position, algo_names, out_fig, title):
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

    Xlabels = {0: "Probability Broken", 1: "Number Demand Pairs", 2: "Demand Flow", 3: "Monitors"}
    datas = np.empty(shape=(len(seeds_values), len(algos), len(X_vals)))

    for ai, al in enumerate(algos):
        algo = al.value[co.AlgoAttributes.NAME]
        for pi, pbro in enumerate(X_vals):
            for si, ss in enumerate(seeds_values):
                if x_position == 0:
                    # varying probs
                    regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-bud|{}-pbro|{}-idv|{}.csv".format(
                        ss,
                        config.graph_dataset.name,
                        config.n_demand_pairs,
                        int(config.demand_capacity),
                        config.supply_capacity,
                        algo,
                        config.monitors_budget,
                        pbro, config.experiment_ind_var.value[0])

                elif x_position == 1:
                    regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-bud|{}-pbro|{}-idv|{}.csv".format(
                        ss,
                        config.graph_dataset.name,
                        pbro,
                        int(config.demand_capacity),
                        config.supply_capacity,
                        algo,
                        config.monitors_budget,
                        config.destruction_quantity, config.experiment_ind_var.value[0])

                elif x_position == 2:
                    # varying flow pp
                    regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-bud|{}-pbro|{}-idv|{}.csv".format(
                        ss,
                        config.graph_dataset.name,
                        config.n_demand_pairs,
                        int(pbro),
                        config.supply_capacity,
                        algo,
                        config.monitors_budget,
                        config.destruction_quantity,
                        config.experiment_ind_var.value[0])

                elif x_position == 3:
                    # varying flow pp
                    regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-bud|{}-pbro|{}-idv|{}.csv".format(
                        ss,
                        config.graph_dataset.name,
                        config.n_demand_pairs,
                        int(config.demand_capacity),
                        config.supply_capacity,
                        algo,
                        pbro,
                        config.destruction_quantity,
                        config.experiment_ind_var.value[0])

                path = path_prefix.format(regex_fname)
                df = pd.read_csv(path)
                df = df[typep]
                datas[si, ai, pi] = df.iloc[0]  # SEED, ALGO, VARIABILE

    plt.figure(figsize=(6, 6))

    # ALGO 1: VAR 1:
    # s1: [1,2,3,4,5]
    # s2: [55,66,77,88]

    for i, algo_en in enumerate(algos):
        avg_val = datas.mean(axis=0)  # ALGO, VAR
        plt.errorbar(X_vals, avg_val[i, :], yerr=sem(datas[:, i, :]), label=algo_names[i],
                     marker=algo_en.value[co.AlgoAttributes.PLOT_MARKER], fillstyle='none')

    plt.legend()
    plt.title(title)
    plt.xlabel(Xlabels[x_position])
    plt.ylabel(plot_name[typep])
    plt.grid(alpha=.4)
    plt.xticks(X_vals, X_vals)
    if typep == "n_monitor_msg":
        plt.yscale('log')

    # print(out)
    # plt.savefig(plot_name[typep] + str(time.time()) + ".png")
    # plt.show()
    plt.tight_layout()
    out_fig.savefig()  # saves the current figure into a pdf page
    plt.close()


def sample_file(seed, graph, np, dc, spc, alg, bud, pbro, indvar):
    return "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-bud|{}-pbro|{}-idv|{}.csv".format(seed, graph, np, dc, spc, alg, bud, pbro, indvar)


def plot_integral(source, config, seeds_values, X_var, algos, plot_type, x_position, outliers:float=0, algo_names=None, out_fig=None, title=None, PERC_DESTRUCTION=None, fixed_x=None):
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
               2: "Demand Flow",
               3: "Monitors"}

    MAX_STEPS = 400
    datas = np.empty(shape=(MAX_STEPS, len(seeds_values), len(algos), len(X_var)))
    datas[:] = np.nan

    # datas fill
    for j, x in enumerate(X_var):
        for i, algo in enumerate(algos):
            algo = algo.value[co.AlgoAttributes.NAME]
            for k, ss in enumerate(seeds_values):
                # varying probability

                if x_position == 0:  # demand capacity
                    MAX_TOTAL_FLOW = np.ones(shape=len(X_var)) * config.n_demand_pairs * config.demand_capacity
                    MAX_FLOW_STEPS = np.ones(shape=len(X_var)) * config.n_demand_pairs * config.demand_capacity
                    regex_fname = sample_file(ss, config.graph_dataset.name, config.n_demand_pairs, int(config.demand_capacity),
                                              config.supply_capacity, algo, config.monitors_budget, x, config.experiment_ind_var.value[0])

                elif x_position == 1:  # demand pairs
                    MAX_TOTAL_FLOW = np.array(X_var) * config.demand_capacity
                    MAX_FLOW_STEPS = np.ones(shape=len(X_var)) * np.array(X_var) * config.demand_capacity
                    regex_fname = sample_file(ss, config.graph_dataset.name, x, int(config.demand_capacity), config.supply_capacity,
                                              algo, config.monitors_budget, config.destruction_quantity, config.experiment_ind_var.value[0])

                elif x_position == 2:  # vary fpp
                    MAX_TOTAL_FLOW = np.array(X_var) * config.n_demand_pairs
                    MAX_FLOW_STEPS = np.ones(shape=len(X_var)) * config.n_demand_pairs * np.array(X_var)
                    regex_fname = sample_file(ss, config.graph_dataset.name, config.n_demand_pairs, int(x), config.supply_capacity,
                                              algo, config.monitors_budget, config.destruction_quantity, config.experiment_ind_var.value[0])

                elif x_position == 3:  # vary monit
                    MAX_TOTAL_FLOW = np.ones(shape=len(X_var)) * config.n_demand_pairs * config.demand_capacity
                    MAX_FLOW_STEPS = np.ones(shape=len(X_var)) * config.n_demand_pairs * config.demand_capacity
                    regex_fname = sample_file(ss, config.graph_dataset.name, config.n_demand_pairs, int(config.demand_capacity), config.supply_capacity, algo, x, config.destruction_quantity, config.experiment_ind_var.value[0])
                else:
                    print("Not handled x_position", x_position)
                    exit()

                path = path_prefix.format(regex_fname)
                df = pd.read_csv(path)["flow_cum"]

                if not(not (algo == "ORACLE" or
                            algo == "CEDAR_MONITOR") or df.iloc[-1] == MAX_TOTAL_FLOW[j]):
                    print("Consider removing seed", ss)

                # assert(not algo == "ORACLE" or df.iloc[-1] == MAX_TOTAL_FLOW[j])
                df_rep = pd.read_csv(path)["repairs"]
                is_rep = 1 - df_rep.isnull() * 1
                is_rep.iloc[-1] = 1
                # print(df)
                # print(is_rep)
                # print(np.where(is_rep > 0))

                df = df[np.where(is_rep > 0)[0]]
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

    RAW_DATA = datas[:]

    # TRUNCATE:  # remove the last lines when all the algos reached the max
    # shape=(MAX_STEPS, len(seeds_values), len(algos), perc_destruction)  > (MAX_STEPS, len(algos), perc_destruction)
    flowz = datas.mean(axis=1)[:, :, :]  # 2D
    flowz_r = np.roll(flowz, -1, axis=0)

    flowz = flowz[:-1, :, :]   # removes the first row due to the shift
    flowz_r = flowz_r[:-1, :, :]

    agreement = (((flowz == flowz_r) * 1).sum(axis=1) == len(algos)) * 1  # 2D [MAX_STEPS, len(X_var)]

    # the index of the last 0 is where they were different for the last time
    agreement_rows = np.where(agreement == 0)[0]
    agreement_cols = np.where(agreement == 0)[1]

    FRONTIER = np.ones(shape=(datas.shape[-1])) * - np.inf
    for i, cc in enumerate(agreement_cols):
        if FRONTIER[cc] < agreement_rows[i]:
            FRONTIER[cc] = agreement_rows[i]
    FRONTIER = FRONTIER + 2

    datas = datas.mean(axis=1)
    for i in range(datas.shape[-1]):  # x
        front = int(FRONTIER[i])
        pad = int(MAX_STEPS - front)
        datas[front:, :, i] = np.zeros(shape=(pad, datas.shape[1]))

    # print(FRONTIER)
    # exit()
    # DONE TRUNCATE

    # sum_flows = (datas.sum(axis=0) / MAX_FLOW_STEPS)
    # avg_sum_flows = sum_flows.mean(axis=0)
    # std_sum_flows = sum_flows.std(axis=0)

    # # REMOVING OUTLIERS
    # av_p_std = avg_sum_flows + std_sum_flows
    # av_m_std = avg_sum_flows - std_sum_flows
    #
    # av_p_std = np.tile(av_p_std, (len(seeds_values), 1, 1))
    # av_m_std = np.tile(av_m_std, (len(seeds_values), 1, 1))
    #
    # A = np.asarray((sum_flows <= av_p_std), dtype=int)
    # B = np.asarray((sum_flows >= av_m_std), dtype=int)
    #
    # THRESHOLD = int(len(algos) * len(X_var) * outliers)
    #
    # # is how many times the seed was in the band
    # normal_seeds = (A*B).sum(axis=(1, 2)) >= THRESHOLD   # a vector as big as the number of seeds, the sum says how many times the samples fall in the stds

    # sum_flows = sum_flows[normal_seeds, :, :]
    # avg_sum_flows = sum_flows.mean(axis=0)
    # std_sum_flows = sum_flows.std(axis=0)

    # print(avg_sum_flows)
    # exit()

    # if outliers > 0:
    #     print("discarded some seeds, now they are", sum_flows.shape[0])
    # # REMOVING OUTLIERS
    #
    # avg_max_flows = (datas.max(axis=0) / MAX_TOTAL_FLOW).mean(axis=0)
    # std_max_flows = (datas.max(axis=0) / MAX_TOTAL_FLOW).std(axis=0)
    #
    # y_plot = avg_max_flows if plot_type == 1 else avg_sum_flows
    # std_y_plot = std_max_flows if plot_type == 1 else std_sum_flows
    #
    plt.figure(figsize=(6, 6))

    # -------------------- PLOT NOW
    if plot_type == 2:
        # shape=(MAX_STEPS, len(seeds_values), len(algos), len(X_var))
        front = int(FRONTIER[PERC_DESTRUCTION])
        avg_flow = datas[:front, :, PERC_DESTRUCTION] / MAX_TOTAL_FLOW[PERC_DESTRUCTION]  # last element
        for i, _ in enumerate(algos):
            plt.plot(np.arange(avg_flow.shape[0]), avg_flow[:, i], label=algo_names[i])
        plt.ylabel("Flow")
        plt.xlabel("Repair Steps")

    elif plot_type == 3:
        ALGO_OUR = 0
        for i in range(len(algos)):
            if i != ALGO_OUR:
                front = int(FRONTIER[PERC_DESTRUCTION])
                A1_avg_flow = datas[:front, ALGO_OUR, PERC_DESTRUCTION] / MAX_TOTAL_FLOW[PERC_DESTRUCTION]
                A2_avg_flow = datas[:front, i, PERC_DESTRUCTION] / MAX_TOTAL_FLOW[PERC_DESTRUCTION]
                out = A1_avg_flow - A2_avg_flow
                label_out = "{} - {}".format(algo_names[ALGO_OUR], algo_names[i])
                plt.plot(np.arange(out.shape[0]), out, label=label_out)
                plt.axhline(y=0, color='r', linestyle=':')
                plt.ylabel("Flow Difference")
                plt.xlabel("Repair Steps")
    else:
        for i, algo_en in enumerate(algos):
            front = FRONTIER
            # np.set_printoptions(precision=2)
            # print(datas.sum(axis=0))
            # print(front)
            # print(front * MAX_FLOW_STEPS)
            # print()

            # if plot_type == 0 and i == 0:
            #     print()
            #     print(datas.sum(axis=0))
            #     print(MAX_FLOW_STEPS)
            #     print(front)
            #     print(front * MAX_FLOW_STEPS)
            #     print((datas.sum(axis=0) / (front * MAX_FLOW_STEPS)))
            #     print()
            # exit()
            avg_sum_flows = (datas.sum(axis=0) / (front * MAX_FLOW_STEPS))
            avg_max_flows = (datas / MAX_TOTAL_FLOW).max(axis=0)   # (numero algo, numero rottura)
            y_plot = avg_max_flows if plot_type == 1 else avg_sum_flows
            ste_y_plot = sem(RAW_DATA.max(axis=0)[:, i, :] / MAX_TOTAL_FLOW) if plot_type == 1 else sem(RAW_DATA.sum(axis=0)[:, i, :] / (front * MAX_FLOW_STEPS))

            # print(X_var, y_plot[i], ste_y_plot)
            # plt.plot(X_var, y_plot[i], label=algo_names[i])
            plt.errorbar(X_var, y_plot[i], yerr=ste_y_plot, label=algo_names[i],
                         marker=algo_en.value[co.AlgoAttributes.PLOT_MARKER], fillstyle='none')
            plt.xticks(X_var)
            # plt.fill_between(X_var, y_plot[i] - std_y_plot[i], y_plot[i] + std_y_plot[i], alpha=0.2)
        plt.ylabel("Total Flow" if plot_type else "Cumulative Flow")
        plt.xlabel(Xlabels[x_position])

    plt.title(title.replace("*", str(fixed_x)) if fixed_x is not None else title)
    plt.legend()
    plt.grid(alpha=.4)
    # plt.savefig("flow" + str(time.time()) + ".png")
    # plt.show()
    plt.tight_layout()
    out_fig.savefig()  # saves the current figure into a pdf page
    plt.close()


def plot_Xvar_Ydems(source, config, seeds_values, X_vals, algos, x_position, n_dem_edges, plot_type, algo_names, out_fig):

    path_prefix = source + "{}"
    Xlabels = {0: "Probability Broken", 1: "Number Demand Pairs", 2: "Demand Flow", 3: "Monitors"}

    MAX_N_REPAIRS = 200
    MAX_N_DEMANDS = max(n_dem_edges)
    datas = np.zeros(shape=(MAX_N_REPAIRS, MAX_N_DEMANDS, len(seeds_values), len(algos), len(X_vals)))

    for ai, al in enumerate(algos):
        algo = al.value[co.AlgoAttributes.NAME]
        for pi, pbro in enumerate(X_vals):
            for si, ss in enumerate(seeds_values):
                if x_position == 0:
                    # varying probs
                    regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-bud|{}-pbro|{}-idv|{}.csv".format(
                        ss,
                        config.graph_dataset.name,
                        config.n_demand_pairs,
                        int(config.demand_capacity),
                        config.supply_capacity,
                        algo,
                        config.monitors_budget,
                        pbro,
                        config.experiment_ind_var.value[0])

                elif x_position == 1:
                    # vary pairs
                    regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-bud|{}-pbro|{}-idv|{}.csv".format(
                        ss,
                        config.graph_dataset.name,
                        pbro,
                        int(config.demand_capacity),
                        config.supply_capacity,
                        algo,
                        config.monitors_budget,
                        config.destruction_quantity,
                        config.experiment_ind_var.value[0])

                elif x_position == 2:
                    # varying flow pp
                    regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-bud|{}-pbro|{}-idv|{}.csv".format(
                        ss,
                        config.graph_dataset.name,
                        config.n_demand_pairs,
                        int(pbro),
                        config.supply_capacity,
                        algo,
                        config.monitors_budget,
                        config.destruction_quantity,
                        config.experiment_ind_var.value[0])

                elif x_position == 3:
                    # varying flow pp
                    regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-bud|{}-pbro|{}-idv|{}.csv".format(
                        ss,
                        config.graph_dataset.name,
                        config.n_demand_pairs,
                        int(config.demand_capacity),
                        config.supply_capacity,
                        algo,
                        pbro,
                        config.destruction_quantity,
                        config.experiment_ind_var.value[0])

                path = path_prefix.format(regex_fname)
                df = pd.read_csv(path)
                df = df.iloc[:, -MAX_N_DEMANDS:]

                df_len = df.shape[0]
                assert (df_len <= MAX_N_REPAIRS)
                # datas = np.empty(shape=(MAX_N_REPAIRS, len(n_dem_edges), len(seeds_values), len(algos), len(X_vals)))
                n_dems = n_dem_edges[pi] if x_position == 1 else n_dem_edges[0]
                datas[:df_len, :n_dems, si, ai, pi] = df.values[:, -n_dems:]

    # avg sees, accumulate flows
    data_cum = np.cumsum(datas, axis=0)   # MAX, DEMS, SEEDS, ALGO, IND_X
    data_cum = np.mean(data_cum, axis=2)  # MAX, DEMS, ALGO, IND_X

    def find_agreement_bottom():
        """ finds the agreement on the cumulative per flow columns """

        flowz_r = np.roll(data_cum, -1, axis=0)
        flowz = data_cum[:-1, :, :, :]  # removes the first row due to the shift
        flowz_r = flowz_r[:-1, :, :, :]
        # print(((flowz == flowz_r) * 1).sum(axis=1))

        agreement = (((flowz == flowz_r) * 1).sum(axis=1) == MAX_N_DEMANDS) * 1  # MAX, ALGO, IND_X
        # the index of the last 0 is where they were different for the last time
        agreement_idx = np.where(agreement == 0)

        FRONTIER = np.ones(shape=agreement.shape[-2:]) * - np.inf
        for i in range(len(agreement_idx[0])):
            if FRONTIER[agreement_idx[1][i], agreement_idx[2][i]] < agreement_idx[0][i]:
                FRONTIER[agreement_idx[1][i], agreement_idx[2][i]] = agreement_idx[0][i]

        # V1 max t, time of satisfaction of the last
        FRONTIER = FRONTIER + 2
        return FRONTIER

    def find_agreement_top():
        max_vec = np.max(data_cum, axis=0)  # max flow routed for each demand DEMS, ALGO, IND_X
        max_vec = np.tile(max_vec, (MAX_N_REPAIRS, 1, 1, 1))  # MAX_N_REPAIRS, DEMS, ALGO, IND_X
        agreement = (data_cum == max_vec) * 1                 # MAX_N_REPAIRS, MAX, ALGO, IND_X
        agreement_idx = np.where(agreement == 0)

        FRONTIER = np.ones(shape=agreement.shape[-3:]) * - np.inf  # DEMS, ALGO, IND_X
        for i in range(len(agreement_idx[0])):
            if FRONTIER[agreement_idx[1][i], agreement_idx[2][i], agreement_idx[3][i]] < agreement_idx[0][i]:
                FRONTIER[agreement_idx[1][i], agreement_idx[2][i], agreement_idx[3][i]] = agreement_idx[0][i]
        FRONTIER = FRONTIER + 2
        return FRONTIER

    plt.figure(figsize=(6, 6))

    if plot_type == 0:
        plot_times = find_agreement_bottom()
        plt.ylabel("Time Last Routing")

    elif plot_type == 1:
        FRONTIER = find_agreement_bottom()
        MAX_FRONTIER = np.max(FRONTIER, axis=0)

        plot_times = np.ones(shape=FRONTIER.shape) * - np.inf
        for i in range(FRONTIER.shape[0]):
            for j in range(FRONTIER.shape[1]):
                front = int(MAX_FRONTIER[j])
                plot_times[i, j] = np.sum(data_cum[:front, :, i, j], axis=(0, 1))
        plt.ylabel("Cumulated Time")

    elif plot_type == 2:
        plot_times = find_agreement_top()
        plot_times = np.average(plot_times, axis=0)
        plt.ylabel("Average Routed Flow")

    else:
        print("plot_type not handled")
        return

    for i, _ in enumerate(algos):
        # plt.plot(X_vals, plot_times[i], label=algo_names[i])
        plt.errorbar(X_vals, plot_times[i], label=algo_names[i], fillstyle='none')
        plt.xticks(X_vals)

    plt.xlabel(Xlabels[x_position])
    plt.legend()
    plt.grid(alpha=.4)
    plt.tight_layout()
    out_fig.savefig()  # saves the current figure into a pdf page
    plt.close()


def plot_Xvar_Ydems2(source, config, seeds_values, X_vals, algos, x_position, n_dem_edges, plot_type, algo_names, out_fig, title):

    path_prefix = source + "{}"
    Xlabels = {0: "Probability Broken", 1: "Number Demand Pairs", 2: "Demand Flow", 3: "Monitors"}

    MAX_N_REPAIRS = 200
    MAX_N_DEMANDS = max(n_dem_edges)
    FRONTIER_IND_X = np.ones(len(X_vals)) * -np.inf

    datas = np.zeros(shape=(MAX_N_REPAIRS, MAX_N_DEMANDS, len(seeds_values), len(algos), len(X_vals)))

    for ai, al in enumerate(algos):
        algo = al.value[co.AlgoAttributes.NAME]
        for pi, pbro in enumerate(X_vals):
            for si, ss in enumerate(seeds_values):
                if x_position == 0:
                    # varying probs
                    regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-bud|{}-pbro|{}-idv|{}.csv".format(
                        ss,
                        config.graph_dataset.name,
                        config.n_demand_pairs,
                        int(config.demand_capacity),
                        config.supply_capacity,
                        algo,
                        config.monitors_budget,
                        pbro,
                        config.experiment_ind_var.value[0])

                elif x_position == 1:
                    # vary pairs
                    regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-bud|{}-pbro|{}-idv|{}.csv".format(
                        ss,
                        config.graph_dataset.name,
                        pbro,
                        int(config.demand_capacity),
                        config.supply_capacity,
                        algo,
                        config.monitors_budget,
                        config.destruction_quantity,
                        config.experiment_ind_var.value[0])

                elif x_position == 2:
                    # varying flow pp
                    regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-bud|{}-pbro|{}-idv|{}.csv".format(
                        ss,
                        config.graph_dataset.name,
                        config.n_demand_pairs,
                        int(pbro),
                        config.supply_capacity,
                        algo,
                        config.monitors_budget,
                        config.destruction_quantity,
                        config.experiment_ind_var.value[0])

                elif x_position == 3:
                    # varying flow pp
                    regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-bud|{}-pbro|{}-idv|{}.csv".format(
                        ss,
                        config.graph_dataset.name,
                        config.n_demand_pairs,
                        int(config.demand_capacity),
                        config.supply_capacity,
                        algo,
                        pbro,
                        config.destruction_quantity,
                        config.experiment_ind_var.value[0])

                path = path_prefix.format(regex_fname)
                df = pd.read_csv(path)
                df = df.iloc[:, -MAX_N_DEMANDS:]

                if FRONTIER_IND_X[pi] < df.shape[0]:
                    FRONTIER_IND_X[pi] = df.shape[0]

                df_len = df.shape[0]
                assert (df_len <= MAX_N_REPAIRS)
                # datas = np.empty(shape=(MAX_N_REPAIRS, len(n_dem_edges), len(seeds_values), len(algos), len(X_vals)))
                n_dems = n_dem_edges[pi] if x_position == 1 else n_dem_edges[0]
                datas[:df_len, :n_dems, si, ai, pi] = df.values[:, -n_dems:]

    # avg sees, accumulate flows
    data_cum = datas  # np.cumsum(datas, axis=0)     # MAX, DEMS, SEEDS, ALGO, IND_X
    # data_cum = np.mean(data_cum, axis=2)  # MAX, DEMS, ALGO, IND_X
    # print(data_cum[:, :,0,3,0])
    # exit()

    plt.figure(figsize=(6, 6))
    plt.ylabel("Cumulated Time")
    sum_flow = np.max(data_cum, axis=0)  # DEMS, SEEDS, ALGO, IND_X
    times = np.argmax(data_cum, axis=0)  # DEMS, SEEDS, ALGO, IND_X
    times = times + (sum_flow == 0) * FRONTIER_IND_X
    times = times + (sum_flow > 0) * 1

    complement = np.zeros(times.shape) + FRONTIER_IND_X
    plot_times = np.sum(complement - times, axis=0) / (FRONTIER_IND_X * data_cum.shape[1])  # SEEDS, ALGO, IND_X
    # print(np.sum(complement - times, axis=0)[0])
    # print((FRONTIER_IND_X * data_cum.shape[1]))
    # print(plot_times[0, :])
    # exit()

    for i, algo_en in enumerate(algos):
        # plt.plot(X_vals, plot_times[i], label=algo_names[i])
        plt.errorbar(X_vals, np.mean(plot_times[:, i, :], axis=0), yerr=sem(plot_times[:, i, :]), label=algo_names[i],
                     marker=algo_en.value[co.AlgoAttributes.PLOT_MARKER], fillstyle='none')  # front: SEEDS, ALGO, IND_X
        plt.xticks(X_vals)

    plt.xlabel(Xlabels[x_position])
    plt.legend()
    plt.title(title)
    plt.grid(alpha=.4)
    plt.tight_layout()
    out_fig.savefig()  # saves the current figure into a pdf page
    plt.close()


def plot_Xflow_Yrepair(source, config, seeds_values, X_var, algos, x_position, fixed_percentage, algo_names, out_fig, title):
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

    MAX_STEPS = 400
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
                                              config.supply_capacity, algo, config.monitors_budget, x, config.experiment_ind_var.value[0])
                elif x_position == 1:
                    MAX_TOTAL_FLOW = int(x * config.demand_capacity)
                    regex_fname = sample_file(ss, config.graph_dataset.name, x, int(config.demand_capacity), config.supply_capacity,
                                              algo, config.monitors_budget, config.destruction_quantity, config.experiment_ind_var.value[0])

                elif x_position == 2:
                    MAX_TOTAL_FLOW = int(config.n_demand_pairs * x)
                    regex_fname = sample_file(ss, config.graph_dataset.name, config.n_demand_pairs, int(x), config.supply_capacity,
                                              algo, config.monitors_budget, config.destruction_quantity, config.experiment_ind_var.value[0])

                elif x_position == 3:
                    MAX_TOTAL_FLOW = int(config.n_demand_pairs * x)
                    regex_fname = sample_file(ss, config.graph_dataset.name, config.n_demand_pairs, int(config.demand_capacity), config.supply_capacity,
                                              algo, x, config.destruction_quantity, config.experiment_ind_var.value[0])

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

    # shape=(NORM_MAX_FLOW_STEPS, len(seeds_values), len(algos), len(X_var))
    avg_sum_flows = data_repairs.mean(axis=1)

    plt.figure(figsize=(6, 6))

    if fixed_percentage:
        avg_sum_flows = avg_sum_flows[:, :, fixed_percentage]  # fix a destruction
        for i, _ in enumerate(algos):
            plt.plot(np.arange(NORM_MAX_FLOW_STEPS)/100, avg_sum_flows[:, i], label=algo_names[i])
        plt.xlabel("Normalized flow ({})".format(MAX_TOTAL_FLOW))
        plt.ylabel("Number of repairs")
    else:
        avg_sum_flows = avg_sum_flows.sum(axis=0)
        for i, _ in enumerate(algos):
            plt.plot(X_var, avg_sum_flows[i], label=algo_names[i])
        plt.xlabel(Xlabels[x_position])
        plt.ylabel("Cumulative Repairs")

    plt.title(title)
    plt.legend()
    plt.grid(alpha=.4)

    out_fig.savefig()  # saves the current figure into a pdf page
    plt.close()


# def intro_nobud():
#     config = ma.setup_configuration()
#     co.PATH_EXPERIMENTS = "data/experiments/"
#
#     dis_uni = {0: [.3, .4, .5, .6, .7, .8],
#                1: .5,
#                2: .5,
#                }
#
#     npairs = {0: 8,
#               1: [5, 6, 7, 8, 9, 10],
#               2: 8,
#               }
#
#     flowpp = {0: 11,
#               1: 11,
#               2: [5, 7, 9, 11, 13, 15],
#               }
#
#     monitor_bud = {0: np.inf,
#                    1: np.inf,
#                    2: np.inf,
#                    }
#
#     ind_var = {0: [co.IndependentVariable.PROB_BROKEN, dis_uni],
#                # 1: [co.IndependentVariable.N_DEMAND_EDGES, npairs],
#                # 2: [co.IndependentVariable.FLOW_DEMAND, flowpp],
#                }
#
#     seeds = set(range(700, 800))
#     seeds -= {700, 701, 703, 705, 714, 717, 721, 720, 722, 724, 726, 731, 736, 738, 740, 741, 744, 748,
#               758, 759, 752, 760, 749, 761, 755, 783, 787, 794, 770, 765, 769, 774, 778, 782, 791, 792, 709, 715, 713}
#     seeds -= {784, 702, 732, 747, 751, 775, 781, 793, 777}  # no limit budget
#     print("Using", len(seeds), seeds)
#
#     BENCHMARKS = [co.Algorithm.TOMO_CEDAR_MONITOR,
#                   co.Algorithm.CEDAR_MONITOR,
#                   co.Algorithm.SHP_MONITOR,
#                   co.Algorithm.ISR_SP_MONITOR, co.Algorithm.ISR_MULTICOM_MONITOR]
#
#     algo_names = [al.value[co.AlgoAttributes.NAME] for al in BENCHMARKS]
#
#     source = co.PATH_EXPERIMENTS
#     OUTLIERS = 0
#
#     return config, dis_uni, npairs, flowpp, seeds, BENCHMARKS, monitor_bud, ind_var, algo_names, source, OUTLIERS


def intro_bud():
    config = ma.setup_configuration()
    co.PATH_EXPERIMENTS = "data/experiments/"

    dis_uni = {0: [.3, .4, .5, .6, .7, .8],
               # 1: .5,
               # 2: .5,
               3: .8
               }

    npairs = {0: 8,
              # 1: [5, 6, 7, 8, 9, 10],
              # 2: 8,
              3: 8
              }

    flowpp = {0: 11,
              # 1: 11,
              # 2: [5, 7, 9, 11, 13, 15],
              3: 11
              }

    monitor_bud = {0: 16,
                   # 1: np.inf,
                   # 2: np.inf,
                   3: [10, 12, 14, 16, 18, 20]
                   }

    ind_var = {0: [co.IndependentVariable.PROB_BROKEN, dis_uni],
               # 1: [co.IndependentVariable.N_DEMAND_EDGES, npairs],
               # 2: [co.IndependentVariable.FLOW_DEMAND, flowpp],
               3: [co.IndependentVariable.MONITOR_BUDGET, monitor_bud]
               }

    seeds = set(range(700, 725))
    seeds -= {700, 701, 703, 705, 714, 717, 721, 720, 722, 724, 726, 731, 736, 738, 740, 741, 744, 748,
              758, 759, 752, 760, 749, 761, 755, 783, 787, 794, 770, 765, 769, 774, 778, 782, 791, 792, 709, 715, 713}
    seeds -= {706, 711, 730, 772, 737, 719, 704, 745, 718, 756, 716}
    print("Using", len(seeds), seeds)

    BENCHMARKS = [co.Algorithm.TOMO_CEDAR, co.Algorithm.ORACLE, co.Algorithm.CEDAR,
                  co.Algorithm.ST_PATH, co.Algorithm.SHP, co.Algorithm.ISR_SP]  # co.Algorithm.ISR_SP,

    algo_names = [al.value[co.AlgoAttributes.NAME] for al in BENCHMARKS]

    source = co.PATH_EXPERIMENTS
    OUTLIERS = 0

    return config, dis_uni, npairs, flowpp, seeds, BENCHMARKS, monitor_bud, ind_var, algo_names, source, OUTLIERS


def plotting_data():

    config, dis_uni, npairs, flowpp, seeds, BENCHMARKS, \
    monitor_bud, ind_var, algo_names, source, OUTLIERS = intro_bud()

    with PdfPages('multipage_pdf.pdf') as pdf:
        for i, (name, vals) in ind_var.items():
            print("Now varying", name.name, "as", vals[i])

            config.supply_capacity = (80, 81)
            PERC_DESTRUCTION = -2

            if name == co.IndependentVariable.PROB_BROKEN:  # vary prob broken fix n_pairs, ffp
                config.experiment_ind_var = co.IndependentVariable.PROB_BROKEN
                config.n_demand_clique = npairs[i]
                config.n_demand_pairs = config.n_edges_given_n_nodes(npairs[i])
                config.demand_capacity = flowpp[i]
                config.monitors_budget = monitor_bud[i]
                fixed_x = dis_uni[i][PERC_DESTRUCTION]
                plot_title = "p_bro-{}|d_node-{}|d_edges-{}|d_cap-{}|m_bud-{}".format("*", config.n_demand_clique, config.n_demand_pairs,
                                                                                      config.demand_capacity, config.monitors_budget)

            elif name == co.IndependentVariable.N_DEMAND_EDGES:  # vary n_pairs fix prob and ffp
                config.experiment_ind_var = co.IndependentVariable.N_DEMAND_EDGES
                config.destruction_quantity = dis_uni[i]
                config.monitors_budget = monitor_bud[i]
                config.demand_capacity = flowpp[i]
                vals[i] = [config.n_edges_given_n_nodes(val) for val in vals[i]]
                fixed_x = npairs[i][PERC_DESTRUCTION]
                plot_title = "p_bro-{}|d_node-{}|d_edges-{}|d_cap-{}|m_bud-{}".format(config.destruction_quantity, config.n_demand_clique, "*",
                                                                                      config.demand_capacity, config.monitors_budget)

            elif name == co.IndependentVariable.FLOW_DEMAND:  # vary fpp fix prob and n_pais
                config.experiment_ind_var = co.IndependentVariable.FLOW_DEMAND
                config.destruction_quantity = dis_uni[i]
                config.n_demand_clique = npairs[i]
                config.n_demand_pairs = config.n_edges_given_n_nodes(npairs[i])
                config.monitors_budget = monitor_bud[i]
                fixed_x = flowpp[i][PERC_DESTRUCTION]
                plot_title = "p_bro-{}|d_node-{}|d_edges-{}|d_cap-{}|m_bud-{}".format(config.destruction_quantity, config.n_demand_clique,
                                                                                      config.n_demand_pairs, "*", config.monitors_budget)

            elif name == co.IndependentVariable.MONITOR_BUDGET:  # vary fpp fix prob and n_pais
                config.experiment_ind_var = co.IndependentVariable.MONITOR_BUDGET
                config.destruction_quantity = dis_uni[i]
                config.n_demand_clique = npairs[i]
                config.n_demand_pairs = config.n_edges_given_n_nodes(npairs[i])
                config.demand_capacity = flowpp[i]
                fixed_x = monitor_bud[i][PERC_DESTRUCTION]
                plot_title = "p_bro-{}|d_node-{}|d_edges-{}|d_cap-{}|m_bud-{}".format(config.destruction_quantity, config.n_demand_clique,
                                                                                      config.n_demand_pairs, config.demand_capacity, "*")

            plot_integral(source, config, seeds, vals[i], BENCHMARKS, plot_type=2, x_position=i, outliers=OUTLIERS, algo_names=algo_names, out_fig=pdf, title=plot_title, PERC_DESTRUCTION=PERC_DESTRUCTION, fixed_x=fixed_x)
            plot_integral(source, config, seeds, vals[i], BENCHMARKS, plot_type=3, x_position=i, outliers=OUTLIERS, algo_names=algo_names, out_fig=pdf, title=plot_title, PERC_DESTRUCTION=PERC_DESTRUCTION, fixed_x=fixed_x)
            plot_integral(source, config, seeds, vals[i], BENCHMARKS, plot_type=1, x_position=i, outliers=OUTLIERS, algo_names=algo_names, out_fig=pdf, title=plot_title, PERC_DESTRUCTION=PERC_DESTRUCTION)
            plot_integral(source, config, seeds, vals[i], BENCHMARKS, plot_type=0, x_position=i, outliers=OUTLIERS, algo_names=algo_names, out_fig=pdf, title=plot_title, PERC_DESTRUCTION=PERC_DESTRUCTION)

            ndmp = vals[i] if i == 1 else [config.n_demand_pairs]
            plot_Xvar_Ydems2(source, config, seeds, vals[i], BENCHMARKS, x_position=i, n_dem_edges=ndmp, plot_type=0,
                             algo_names=algo_names, out_fig=pdf, title=plot_title)


            plot_monitors_stuff(source, config, seeds, vals[i], BENCHMARKS, typep="n_repairs", x_position=i, algo_names=algo_names, out_fig=pdf, title=plot_title)
            # plot_monitors_stuff(source, config, seeds, vals[i], BENCHMARKS, typep="n_monitor_msg", x_position=i, algo_names=algo_names, out_fig=pdf, title=plot_title)
            plot_monitors_stuff(source, config, seeds, vals[i], BENCHMARKS, typep="n_monitors", x_position=i, algo_names=algo_names, out_fig=pdf, title=plot_title)



if __name__ == '__main__':
    plotting_data()
