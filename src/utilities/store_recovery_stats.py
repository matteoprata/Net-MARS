
import os
import numpy as np

import src.constants as co
import pandas as pd
from src.recovery_protocols.protocols.MinTDS import MinTDS
from src.recovery_protocols.protocols.PRoTOnDyn import PRoTOnDyn


def save_stats_dynamic(stats, fname, algon):
    df = stats.df

    print("saving stats > {}".format(fname))
    if os.path.exists(co.PATH_EXPERIMENTS):
        df.to_csv("{}{}".format(co.PATH_EXPERIMENTS, fname))
    else:
        os.makedirs(co.PATH_EXPERIMENTS)
        df.to_csv("{}{}".format(co.PATH_EXPERIMENTS, fname))
    return df


def save_stats_monotonous(stats, fname, algon):
    """ saving number of repairs and flow routed """

    print("merging stats for output file")
    for i, j in enumerate(stats):
        print("\n", i, j)

    repairs, iter, flow_cum, is_forced_tot, is_total_flow = [], [], [], [], []
    n_repairs = 0
    demand_pairs = {k: [] for k in stats[-1]["demands_sat"].keys()}
    for i, dic in enumerate(stats):  # iteration index
        purge = ["None"] if algon == co.Algorithm.MIN_TDS else []  # for the algorithms with no final empy iteration # TODO improve integration
        vals = dic["node"] + dic["edge"] + purge
        # number of repairs in this iteration, to propagate values accordingly
        n_vals = max(len(vals), 1)   # last iteration has 0 repairs and counts as a None repair
        # print("n_vals", n_vals)
        repairs += vals if len(vals) > 0 else [None]
        iter += [dic["iter"]] * n_vals
        n_repairs += len(vals) - len(purge)

        if algon == co.Algorithm.MIN_TDS:
            # CUMULATIVE FLOW
            i_demand_pairs = stats[i]["demands_sat"] if "demands_sat" in stats[i].keys() else []
            cumulative_flow = None
            for k in i_demand_pairs:
                demand_pairs[k] = i_demand_pairs[k]
                if cumulative_flow is None:
                    cumulative_flow = np.cumsum(i_demand_pairs[k])
                else:
                    cumulative_flow += np.cumsum(i_demand_pairs[k])
            flow_cum = cumulative_flow

        elif algon == co.Algorithm.PROTON_DYN:
            # CUMULATIVE FLOW
            flow_cum += [stats[i]["flow"]] * n_vals

            # is_forced = stats[i]["event"] == "destroy"
            # d_flow = [0] * n_vals
            # d_flow[0] = 1 if is_forced else 0
            events = ["none"] * n_vals
            events[0]  = stats[i]["event"] if stats[i]["event"] == "destroy" else events[0]
            events[-1] = stats[i]["event"] if stats[i]["event"] != "destroy" else events[-1]

            tot_flow = [stats[i]["total_flow"]] * n_vals
            is_forced_tot += events  # d_flow
            is_total_flow += tot_flow
        else:
            # CUMULATIVE FLOW
            flow_cum += [stats[i]["flow"]] * n_vals

            # [0000 30 0]
            i_demand_pairs = stats[i]["demands_sat"] if "demands_sat" in stats[i].keys() else []
            for k in i_demand_pairs:
                # print(k, i_demand_pairs)
                d_flow = [0] * n_vals
                d_flow[0] = stats[i]["demands_sat"][k][i]
                demand_pairs[k] += d_flow

    print(demand_pairs)
    print(len(iter), len(repairs), flow_cum)
    df = pd.DataFrame()
    df["repairs"] = repairs
    df["iter"] = iter
    df["flow_cum"] = flow_cum

    # position 0 we set the number of repairs
    none_vec = [None]*len(iter)

    n_repairs_vector = none_vec[:]
    n_repairs_vector[0] = n_repairs
    df["n_repairs"] = n_repairs_vector

    n_monitors_vector = none_vec[:]
    n_monitors_vector[0] = len(stats[-1]["monitors"])
    df["n_monitors"] = n_monitors_vector

    n_monitor_msg_messages = none_vec[:]
    n_monitor_msg_messages[0] = stats[-1]["packet_monitoring"]  # packet_monitor
    df["n_monitor_msg"] = n_monitor_msg_messages
    secs = none_vec[:]

    if "execution_sec" in stats[-1].keys():
        secs[0] = stats[-1]["execution_sec"]
        df["execution_sec"] = secs

    if algon == co.Algorithm.PROTON_DYN:
        df["event"] = is_forced_tot
        df["total_flow"] = is_total_flow
    else:
        for k in demand_pairs:
            df["d-" + str(k)] = demand_pairs[k]

    print("saving stats > {}".format(fname))
    if os.path.exists(co.PATH_EXPERIMENTS):
        df.to_csv("{}{}".format(co.PATH_EXPERIMENTS, fname))
    else:
        os.makedirs(co.PATH_EXPERIMENTS)
        df.to_csv("{}{}".format(co.PATH_EXPERIMENTS, fname))
    return df


def save_stats_NON_monotonous(stats, fname):
    """ saving number of repairs and flow routed """

    for i in stats:
        print(i)

    i_demand_pairs = stats[-1]["demands_sat"] if "demands_sat" in stats[-1].keys() else []
    stopz = dict()
    for k in i_demand_pairs:  # iterates demand edges
        stop = len(i_demand_pairs[k]) - 1  # iteration indices
        for ite_flow in reversed(i_demand_pairs[k]):
            if ite_flow != i_demand_pairs[k][-1]:  # different from max_flow
                break
            stop -= 1
        stopz[k] = stop+1
        # print("ECCO", k, i_demand_pairs[k], stop)

    repairs, iter, flow_cum = [], [], []
    n_repairs = 0
    demand_pairs = {k: [] for k in stats[-1]["demands_sat"].keys()}
    for i, dic in enumerate(stats):  # iteration index
        vals = dic["node"] + dic["edge"]
        # numbers in this iteration, to propagate values accordingly
        n_vals = max(len(vals), 1)
        repairs += vals if len(vals) > 0 else [None]
        iter += [dic["iter"]] * n_vals
        flow_cum += [stats[i]["flow"]] * n_vals  # IGNORED
        n_repairs += len(vals)

        i_demand_pairs = stats[i]["demands_sat"] if "demands_sat" in stats[i].keys() else []
        for k in i_demand_pairs:  # iterates demand edges
            d_flow = [0] * n_vals
            d_flow[0] = stats[i]["demands_sat"][k][i] if i == stopz[k] else 0
            demand_pairs[k] = demand_pairs[k] + d_flow

    df = pd.DataFrame()
    df["repairs"] = repairs
    df["iter"] = iter
    # df["flow_cum"] = flow_cum

    flows = np.array([i for i in demand_pairs.values()]).T
    df["flow_cum"] = np.sum(np.cumsum(flows, axis=0), axis=1)

    # position 0 we set the number of repairs
    none_vec = [None]*len(flow_cum)

    n_repairs_vector = none_vec[:]
    n_repairs_vector[0] = n_repairs
    df["n_repairs"] = n_repairs_vector

    n_monitors_vector = none_vec[:]
    n_monitors_vector[0] = len(stats[-1]["monitors"])
    df["n_monitors"] = n_monitors_vector

    n_monitor_msg_messages = none_vec[:]
    n_monitor_msg_messages[0] = stats[-1]["packet_monitoring"]  # packet_monitor
    df["n_monitor_msg"] = n_monitor_msg_messages

    for k in demand_pairs:
        df["d-" + str(k)] = demand_pairs[k]

    if os.path.exists(co.PATH_EXPERIMENTS):
        df.to_csv("{}{}".format(co.PATH_EXPERIMENTS, fname))
    else:
        os.makedirs(co.PATH_EXPERIMENTS)
        df.to_csv("{}{}".format(co.PATH_EXPERIMENTS, fname))
    return df