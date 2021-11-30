import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

def save_stats_as_df_ph1(stats, config):
    """ saving number of repairs and flow routed """

    repairs, iter, flow_cum, flow_perc = [], [], [], []
    routed_flow = 0
    for i, dic in enumerate(stats):
        vals = dic["node"] + dic["edge"]
        n_vals = max(len(vals), 1)
        repairs += vals if len(vals) > 0 else [None]
        iter += [dic["iter"]]*n_vals
        flow_perc += [stats[i-1]["flow"]]*n_vals if i > 0 else [0]*n_vals

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
    fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}.csv".format(config.seed, config.graph_dataset.name, config.n_demand_pairs,
                                                                 config.demand_capacity, config.supply_capacity[0], config.algo_name)
    print("saving stats > {}".format(fname))
    df.to_csv("data/experiments/{}".format(fname))


def average_flow_seeds_ph2(config):
    """ given multiple stats files with a regular expression, averages the flows """

    path_prefix = "data/experiments/{}"
    regex_fname = "exp-s|*-g|{}-np|{}-dc|{}-spc|{}-alg|{}.csv".format(config.graph_dataset.name, config.n_demand_pairs, config.demand_capacity,
                                                                      config.supply_capacity[0], config.algo_name)
    path = path_prefix.format(regex_fname)
    list_files = glob.glob(path)
    print("looking for path:", path)

    stats_shape = []
    index_shape = []
    for i, fname in enumerate(list_files):
        df = pd.read_csv(fname)
        stats_shape.append(df.shape[0])
        index_shape.append(i)
    ind_max_shape = np.argmax(stats_shape)    # the longest df
    start_longest = index_shape[ind_max_shape]

    fill = pd.DataFrame()
    fill["flow0"] = pd.read_csv(list_files[start_longest])["flow_cum"]
    del list_files[start_longest]

    for i, fname in enumerate(list_files):
        fill["flow{}".format(i+1)] = pd.read_csv(fname)["flow_cum"]

    fill = fill.fillna(method="ffill")
    fill["avg_flow"] = fill.mean(axis=1)
    fname = "avg-flow-s|W-g|{}-np|{}-dc|{}-spc|{}-alg|{}.csv".format(config.graph_dataset.name, config.n_demand_pairs,
                                                                     config.demand_capacity, config.supply_capacity, config.algo_name)
    fill.to_csv(path_prefix.format(fname))


def average_repairs_seeds_ph2(config):
    """ given multiple stats files with a regular expression, averages the repairs """

    path_prefix = "data/experiments/{}"
    regex_fname = "exp-s|*-g|{}-np|{}-dc|{}-spc|{}-alg|{}.csv".format(config.graph_path, config.n_demand_pairs,
                                                                      config.demand_capacity, config.supply_capacity, config.algo_name)
    path = path_prefix.format(regex_fname)
    list_files = glob.glob(path)

    fill = pd.DataFrame()
    for i, fname in enumerate(list_files):
        fill["rep-count{}".format(i)] = [pd.read_csv(fname).shape[0]]

    fill["avg_rep-count"] = fill.mean(axis=1)
    print(fill.to_string())
    fname = "avg-repairs-s|W-g|{}-np|{}-dc|{}-spc|{}-alg|{}.csv".format(config.graph_dataset.name, config.n_demand_pairs,
                                                                        config.demand_capacity, config.supply_capacity, config.algo_name)
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
    fill[algo_name] = pd.read_csv(list_files[start_longest])["avg_flow"]
    del list_files[start_longest]

    for i, fname in enumerate(list_files):
        algo_name = fname.split("alg|")[1].split(".csv")[0]
        fill[algo_name] = pd.read_csv(fname)["avg_flow"]

    fill = fill.fillna(method="ffill")
    fill.plot(xlabel="Time", ylabel="Flow (Cumulative)", title="D-Edges n:5 flow:15u | S-Edges flow:30u | uniform 50% disruption.")
    plt.show()


def aggregate_multiple_seeds(config):
    average_flow_seeds_ph2(config)
    aggregate_average_plot_ph3(config)
    # average_repairs_seeds_ph2(config)


