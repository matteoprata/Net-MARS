import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from collections import defaultdict
# repairs iter  flow_cum  flow_perc  n_repairs  n_monitors  n_monitor_msg


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

    fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-pbro|{}.csv".format(config.seed, config.graph_dataset.name, config.n_demand_pairs,
                                                                 config.demand_capacity, config.supply_capacity, config.algo_name, config.destruction_uniform_quantity)
    print("saving stats > {}".format(fname))
    df.to_csv("data/experiments/{}".format(fname))


def average_flow_seeds_ph2(config):
    """ given multiple stats files with a regular expression, averages the flows """
    average_columns(config, "flow_cum", "flow")


def average_repair_seeds_ph2(config):
    """ given multiple stats files with a regular expression, averages the flows """
    average_columns(config, "n_repairs", "repairs")


def average_monitor_seeds_ph2(config):
    """ given multiple stats files with a regular expression, averages the flows """
    average_columns(config, "n_monitors", "monitors")


def average_messages_seeds_ph2(config):
    """ given multiple stats files with a regular expression, averages the flows """
    average_columns(config, "n_monitor_msg", "monitormsg")


def average_columns(config, column, fname_val):
    path_prefix = "data/experiments/{}"
    regex_fname = "exp-s|*-g|{}-np|{}-dc|{}-spc|{}-alg|{}.csv".format(config.graph_dataset.name, config.n_demand_pairs,
                                                                      config.demand_capacity,
                                                                      config.supply_capacity, config.algo_name)
    path = path_prefix.format(regex_fname)
    list_files = glob.glob(path)
    print("looking for path:", path)

    stats_shape = []
    index_shape = []
    for i, fname in enumerate(list_files):
        df = pd.read_csv(fname)
        stats_shape.append(df.shape[0])
        index_shape.append(i)
    ind_max_shape = np.argmax(stats_shape)  # the longest df
    start_longest = index_shape[ind_max_shape]

    fill = pd.DataFrame()
    fill["flow0"] = pd.read_csv(list_files[start_longest])[column]
    del list_files[start_longest]

    for i, fname in enumerate(list_files):
        fill["flow{}".format(i + 1)] = pd.read_csv(fname)[column]

    fill = fill.fillna(method="ffill")
    fill["avg"] = fill.mean(axis=1)
    fname = "avg-{}-s|W-g|{}-np|{}-dc|{}-spc|{}-alg|{}.csv".format(fname_val, config.graph_dataset.name, config.n_demand_pairs,
                                                                     config.demand_capacity, config.supply_capacity,
                                                                     config.algo_name)
    fill.to_csv(path_prefix.format(fname))


def aggregate_average_plot_ph3(config):
    path_prefix = "data/experiments/{}"
    regex_fname = "avg-flow-s|W-g|{}-np|{}-dc|{}-spc|{}-alg|*.csv".format(config.graph_dataset.name, config.n_demand_pairs,
                                                                      config.demand_capacity,
                                                                      config.supply_capacity)
    path = path_prefix.format(regex_fname)
    print("looking for path:", path)

    list_files = glob.glob(path)
    print(list_files)

    stats_shape = []
    index_shape = []
    for i, fname in enumerate(list_files):
        df = pd.read_csv(fname)
        stats_shape.append(df.shape[0])
        index_shape.append(i)
    ind_max_shape = np.argmax(stats_shape)  # the longest df
    start_longest = index_shape[ind_max_shape]

    algo_name = list_files[start_longest].split("alg|")[1].split(".csv")[0]
    fill = pd.DataFrame()
    fill[algo_name] = pd.read_csv(list_files[start_longest])["avg"]
    del list_files[start_longest]

    for i, fname in enumerate(list_files):
        algo_name = fname.split("alg|")[1].split(".csv")[0]
        fill[algo_name] = pd.read_csv(fname)["avg"]

    fill = fill.fillna(method="ffill")
    fill.plot(xlabel="Time", ylabel="Flow (Cumulative)", title="D-Edges n:5 flow:15u | S-Edges flow:30u | uniform 50% disruption.")
    plt.show()


def plot_ph3(config, type):
    path_prefix = "data/experiments/{}"
    regex_fname = "avg-{}-s|W-g|{}-np|{}-dc|{}-spc|{}-alg|*.csv".format(type, config.graph_dataset.name, config.n_demand_pairs,
                                                                        config.demand_capacity,
                                                                        config.supply_capacity)

    path = path_prefix.format(regex_fname)
    print("looking for path:", path)

    list_files = glob.glob(path)
    print(list_files)

    stats_shape = []
    index_shape = []
    for i, fname in enumerate(list_files):
        df = pd.read_csv(fname)
        stats_shape.append(df.shape[0])
        index_shape.append(i)
    ind_max_shape = np.argmax(stats_shape)  # the longest df
    start_longest = index_shape[ind_max_shape]

    algo_name = list_files[start_longest].split("alg|")[1].split(".csv")[0]
    fill = pd.DataFrame()
    fill[algo_name] = pd.read_csv(list_files[start_longest])["avg"]
    del list_files[start_longest]

    for i, fname in enumerate(list_files):
        algo_name = fname.split("alg|")[1].split(".csv")[0]
        fill[algo_name] = pd.read_csv(fname)["avg"]

    fill = fill.iloc[1, :]
    if type == "monitormsg":
        ax = fill.plot.bar(rot=0, log=True)
    else:
        ax = fill.plot.bar(rot=0, log=False)

    ax.set_ylabel(type)
    plt.show()


def plot_monitors_stuff(source, config, seeds_values, pbro_values, algos, typep):
    plot_name = {"n_monitors": "Number monitors", "n_monitor_msg": "Number monitoring messages", "n_repairs": "Number repairs"}
    path_prefix = source + "{}"

    slices = []
    for al in algos:
        seeds_pbro = []
        for pbro in pbro_values:
            dfs = []  # many dfs as the seeds (columns to average on axis 0)
            for ss in seeds_values:
                regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-pbro|{}.csv".format(ss, config.graph_dataset.name,
                                                                                           config.n_demand_pairs,
                                                                                           config.demand_capacity,
                                                                                           config.supply_capacity,
                                                                                           al, pbro)
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
        plt.plot(pbro_values, out, label=algos[i])

    plt.legend()
    plt.xlabel("Probability Broken")
    plt.ylabel(plot_name[typep])
    plt.grid(alpha=.4)
    plt.xticks(pbro_values, pbro_values)
    if typep == "n_monitor_msg":
        plt.yscale('log')

    # print(out)
    plt.show()
    plt.savefig(path_prefix.format(typep + ".png"), dpi=400)
    plt.clf()


def plot_integral(source, config, seeds_values, pbro_values, algos):
    path_prefix = source + "{}"  # "data/experiments/{}"

    dfones = []
    dftwos = []
    for pbro in pbro_values:
        seeds_pbro = []
        for i, algo in enumerate(algos):
            dfs = []  # many dfs as the seeds (columns to average on axis 0)
            for ss in seeds_values:
                regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-pbro|{}.csv".format(ss, config.graph_dataset.name,
                                                                                           config.n_demand_pairs,
                                                                                           config.demand_capacity,
                                                                                           config.supply_capacity,
                                                                                           algo, pbro)
                path = path_prefix.format(regex_fname)
                df = pd.read_csv(path)["flow_cum"]
                dfs.append(df)

            res = pd.concat(dfs, axis=1)
            res.columns = seeds_values
            seeds_pbro.append(res)

        # extending the integral
        assert (len(algos) == 2)
        out_df1 = []
        out_df2 = []
        for se in seeds_values:
            arte = pd.concat([seeds_pbro[0].loc[:, se].dropna(), seeds_pbro[1].loc[:, se].dropna()], axis=1).fillna(method="ffill")
            out_df1.append(arte.iloc[:,0])
            out_df2.append(arte.iloc[:,1])

        out_df1 = pd.concat(out_df1, axis=1).fillna(method="ffill")#.sum(axis=0).T
        out_df2 = pd.concat(out_df2, axis=1).fillna(method="ffill")#.sum(axis=0).T

        dfones.append(out_df1)
        dftwos.append(out_df2)

    A1 = pd.concat(dfones, axis=1).fillna(method="ffill").sum(axis=0)
    A2 = pd.concat(dftwos, axis=1).fillna(method="ffill").sum(axis=0)

    A1 = pd.DataFrame(A1.values.reshape(len(seeds_values), len(pbro_values), order='F'))
    A2 = pd.DataFrame(A2.values.reshape(len(seeds_values), len(pbro_values), order='F'))

    # extend the integral!
    assert (len(algos) == 2)
    positions = [np.array(range(len(pbro_values))) * 2.0 - 0.4, np.array(range(len(pbro_values)))*2.0+0.4]
    colors = ['#D7191C', '#2C7BB6']

    ALG = [A1, A2]
    for i, seeds_pbro in enumerate(ALG):
        seeds_pbro.columns = pbro_values
        bpl = plt.boxplot(seeds_pbro, positions=positions[i])
        plt.plot([], c=colors[i], label=algos[i])

        plt.setp(bpl['boxes'], color=colors[i])
        # plt.setp(bpl['whiskers'], color=colors[i])
        # plt.setp(bpl['caps'], color=colors[i])
        # plt.setp(bpl['medians'], color=colors[i])

    plt.legend()
    plt.xticks(range(0, len(pbro_values) * 2, 2), pbro_values)
    plt.xlabel("Probability Broken")
    plt.ylabel("Cumulative Flow")
    plt.grid(alpha=.4)
    plt.show()
    plt.savefig(path_prefix.format("integral_flow_box.png"), dpi=400)
    plt.clf()

    # >>>>> NOW MEANS <<<<<<
    for i, seeds_pbro in enumerate(ALG):
        seeds_pbro.columns = pbro_values
        mev = seeds_pbro.mean(axis=0)
        stv = seeds_pbro.std(axis=0)

        plt.plot(pbro_values, mev)
        plt.fill_between(pbro_values, mev-stv, mev+stv, alpha=0.2, label=algos[i])

    plt.legend()
    plt.xticks(pbro_values, pbro_values)
    plt.xlabel("Probability Broken")
    plt.ylabel("Cumulative Flow")
    plt.grid(alpha=.4)
    plt.show()
    plt.savefig(path_prefix.format("integral_flow_curves.png"), dpi=400)
    plt.clf()


def plot_monitors(config, seeds_values, pbro_values):
    path_prefix = "data/experiments/{}"

    seeds_pbro = []
    for pbro in pbro_values:
        dfs = []  # many dfs as the seeds (columns to average on axis 0)
        for ss in seeds_values:
            regex_fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-pbro|{}.csv".format(ss, config.graph_dataset.name, config.n_demand_pairs,
                                                                      config.demand_capacity,
                                                                      config.supply_capacity, config.algo_name, pbro)
            path = path_prefix.format(regex_fname)

            df = pd.read_csv(path)["n_monitors"]  # n_monitor_msg
            df = pd.DataFrame([df.iloc[0]])
            dfs.append(df)

        seed_df = pd.concat(dfs, axis=1).T
        seeds_pbro.append(seed_df)

    out = pd.concat(seeds_pbro, axis=1)
    out.columns = pbro_values
    ax = out.boxplot()

    ax.set_xlabel("Probability Broken")
    ax.set_ylabel("Number Monitors")
    ax.grid(alpha=.4)

    print(out)
    plt.show()


def aggregate_multiple_seeds(config):
    average_flow_seeds_ph2(config)
    average_repair_seeds_ph2(config)
    average_messages_seeds_ph2(config)
    average_monitor_seeds_ph2(config)

    aggregate_average_plot_ph3(config)
    plot_ph3(config, "repairs")
    plot_ph3(config, "monitors")
    plot_ph3(config, "monitormsg")



