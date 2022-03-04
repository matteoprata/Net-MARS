import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from collections import defaultdict
# repairs iter  flow_cum  flow_perc  n_repairs  n_monitors  n_monitor_msg
import random


def save_stats_as_df_ph1(stats, config):
    """ saving number of repairs and flow routed """

    repairs, iter, flow_cum, flow_perc = [], [], [], []
    n_repairs = 0
    for i, dic in enumerate(stats):
        vals = dic["node"] + dic["edge"]
        n_vals = max(len(vals), 1)
        repairs += vals if len(vals) > 0 else [None]
        iter += [dic["iter"]]*n_vals
        flow_perc += [stats[i-1]["flow"]]*n_vals if i > 0 else [0]*n_vals
        n_repairs += len(vals)

    flow_perc[-1] = stats[-1]["flow"]
        # for every flow increase, find expected routed flow for the path
        # for _ in range(n_vals):
        #     routed_flow += dic["flow"] / n_vals
        #     flow_cum += [routed_flow]

    df = pd.DataFrame()
    df["repairs"] = repairs
    df["iter"] = iter
    df["flow_cum"] = flow_perc
    df["flow_perc"] = flow_perc

    # position 0 we set the number of repairs
    none_vec = [None]*len(flow_perc)

    n_repairs_vector = none_vec[:]
    n_repairs_vector[0] = n_repairs
    df["n_repairs"] = n_repairs_vector

    n_monitors_vector = none_vec[:]
    n_monitors_vector[0] = len(stats[-1]["monitors"])
    df["n_monitors"] = n_monitors_vector

    n_monitor_msg_messages = none_vec[:]
    n_monitor_msg_messages[0] = stats[-1]["packet_monitoring"]  # packet_monitor
    df["n_monitor_msg"] = n_monitor_msg_messages

    key = "_BUD_" + str(config.monitors_budget)
    fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-pbro|{}.csv".format(config.seed, config.graph_dataset.name, config.n_demand_clique,
                                                                         config.demand_capacity, config.supply_capacity, config.algo_name + key, config.destruction_quantity)
    print("saving stats > {}".format(fname))
    df.to_csv("data/experiments/{}".format(fname))


def plot_monitors_stuff(source, config, seeds_values, X_vals, algos, typep, x_position):
    """

    :param source:
    :param config:
    :param seeds_values:
    :param X_vals:
    :param algos:
    :param typep:
    :return:
    """
    plot_name = {"n_monitors": "Number monitors", "n_monitor_msg": "Number monitoring messages", "n_repairs": "Number repairs"}
    path_prefix = source + "{}"
    Xlabels = {0:"Probability Broken", 1:"Number Demand Nodes", 2:"Demand Flow"}

    slices = []
    for al in algos:
        seeds_pbro = []
        for pbro in X_vals:
            dfs = []  # many dfs as the seeds (columns to average on axis 0)
            for ss in seeds_values:

                if x_position == 0:
                    # varying probs
                    regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-pbro|{}.csv".format(ss, config.graph_dataset.name,
                                                                                               config.n_demand_pairs,
                                                                                               int(config.demand_capacity),
                                                                                               config.supply_capacity,
                                                                                               al, pbro)
                elif x_position == 1:
                    regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-pbro|{}.csv".format(ss,
                                                                                               config.graph_dataset.name,
                                                                                               pbro,
                                                                                               int(config.demand_capacity),
                                                                                               config.supply_capacity,
                                                                                               al, 0.3)

                elif x_position == 2:
                    # varying flow pp
                    regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-pbro|{}.csv".format(ss,
                                                                                               config.graph_dataset.name,
                                                                                               8,
                                                                                               pbro,
                                                                                               config.supply_capacity,
                                                                                               al, 0.3)

                path = path_prefix.format(regex_fname)

                df = pd.read_csv(path)
                df = df[typep]
                df = pd.DataFrame([df.iloc[0]])
                dfs.append(df)

            seed_df = pd.concat(dfs, axis=1).T
            seeds_pbro.append(seed_df)
        slices.append(seeds_pbro)

    for i, sli in enumerate(slices):
        out = pd.concat(sli, axis=1).mean(axis=0)
        # out.columns = pbro_values
        # out = out
        # print(out)
        # exit()
        plt.plot(X_vals, out, label=algos[i])

    plt.legend()
    plt.xlabel(Xlabels[x_position])
    plt.ylabel(plot_name[typep])
    plt.grid(alpha=.4)
    plt.xticks(X_vals, X_vals)
    if typep == "n_monitor_msg":
        plt.yscale('log')

    # print(out)
    plt.savefig(path_prefix.format(typep + ".png"), dpi=400)
    plt.show()
    plt.clf()


def plot_integral(source, config, seeds_values, X_var, algos, is_total, x_position):
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
    Xlabels = {0:"Probability Broken", 1:"Number Demand Nodes", 2:"Demand Flow"}

    algos_values = [[] for _ in range(len(algos))]
    for x in X_var:
        seeds_pbro = []
        for i, algo in enumerate(algos):
            dfs = []  # many dfs as the seeds (columns to average on axis 0)
            for ss in seeds_values:
                # varying probability
                if x_position == 0:
                    regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-pbro|{}.csv".format(ss, config.graph_dataset.name,
                                                                                               config.n_demand_pairs,
                                                                                               int(config.demand_capacity),
                                                                                               config.supply_capacity,
                                                                                               algo, x)
                elif x_position == 1:
                    # # varying n pairs
                    regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-pbro|0.3.csv".format(ss, config.graph_dataset.name,
                                                                                               x,
                                                                                               int(config.demand_capacity),
                                                                                               config.supply_capacity,
                                                                                               algo)

                elif x_position == 2:
                    regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-pbro|{}.csv".format(ss,
                                                                                                config.graph_dataset.name,
                                                                                                8,
                                                                                                x,
                                                                                                config.supply_capacity,
                                                                                                algo, 0.3)

                path = path_prefix.format(regex_fname)
                df = pd.read_csv(path)["flow_cum"]
                dfs.append(df)

            res = pd.concat(dfs, axis=1)
            res.columns = seeds_values
            seeds_pbro.append(res)

        # extending the integral
        algos_values_in = [[] for _ in range(len(algos))]
        for se in seeds_values:
            to_conc = []
            for i in range(len(algos)):
                to_conc.append(seeds_pbro[i].loc[:, se].dropna())

            arte = pd.concat(to_conc, axis=1).fillna(method="ffill")
            for i in range(len(algos)):
                algos_values_in[i].append(arte.iloc[:, i])

        for i in range(len(algos)):
            algos_values_in[i] = pd.concat(algos_values_in[i], axis=1).fillna(method="ffill")

        for i in range(len(algos)):
            algos_values[i].append(algos_values_in[i])

    print(algos_values[0])

    if is_total:
        A = [pd.concat(algos_values[i], axis=1).fillna(method="ffill").max(axis=0) for i in range(len(algos))]
    else:
        A = [pd.concat(algos_values[i], axis=1).fillna(method="ffill").sum(axis=0) for i in range(len(algos))]
    A = [pd.DataFrame(A[i].values.reshape(len(seeds_values), len(X_var), order='F')) for i in range(len(algos))]

    ALG = A
    # extend the integral!
    # positions = [np.array(range(len(pbro_values))) * 2.0 - 0.4,
    #              np.array(range(len(pbro_values))) * 2.0 + 0.4]
    # colors = [(random.random(), random.random(), random.random())]*len(algos)
    #
    #
    # for i, seeds_pbro in enumerate(ALG):
    #     seeds_pbro.columns = pbro_values
    #     bpl = plt.boxplot(seeds_pbro, positions=positions[i])
    #     plt.plot([], c=colors[i], label=algos[i])
    #
    #     plt.setp(bpl['boxes'], color=colors[i])
    #     # plt.setp(bpl['whiskers'], color=colors[i])
    #     # plt.setp(bpl['caps'], color=colors[i])
    #     # plt.setp(bpl['medians'], color=colors[i])
    #
    # plt.legend()
    # plt.xticks(range(0, len(pbro_values) * 2, 2), pbro_values)
    # plt.xlabel("Probability Broken")
    # plt.ylabel("Cumulative Flow")
    # plt.grid(alpha=.4)
    # plt.show()
    # plt.savefig(path_prefix.format("integral_flow_box.png"), dpi=400)
    # plt.clf()

    # >>>>> NOW MEANS <<<<<<
    for i, seeds_pbro in enumerate(ALG):
        seeds_pbro.columns = X_var
        mev = seeds_pbro.mean(axis=0)
        stv = seeds_pbro.std(axis=0)

        plt.plot(X_var, mev)
        plt.fill_between(X_var, mev - stv, mev + stv, alpha=0.2, label=algos[i])

    plt.legend()
    plt.xticks(X_var, X_var)
    plt.xlabel(Xlabels[x_position])
    plt.ylabel("Total Flow" if is_total else "Cumulative Flow")
    plt.grid(alpha=.4)
    plt.savefig(path_prefix.format("integral_flow_curves_{}.png".format(int(is_total))), dpi=400)
    plt.show()
    plt.clf()


