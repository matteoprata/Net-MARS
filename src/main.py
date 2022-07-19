
import numpy as np
import itertools
import argparse
from multiprocessing import Pool
import os
import pandas as pd
import traceback
import signal
import src.constants as co
import src.configuration as configuration
import src.recovery_protocols.main_tomocedar_setup as main_tomocedar_setup
import src.recovery_protocols.main_ISR_setup as main_stocedar_setup
import src.recovery_protocols.main_cedar_setup as main_cedar_setup
import src.recovery_protocols.main_SHP_setup as main_shp_setup

from src.utilities.util import set_seeds, disable_print, enable_print
import src.utilities.util as util
import time

original_config = configuration.Configuration()
time_batch_exec = time.strftime("%Y-%m-%d_%H-%M")

# -----> BEGIN of variable parameters

parser = argparse.ArgumentParser(description='Tomo Cedar recovery algorithm run parameters.')
parser.add_argument('-s',  '--seed', type=int, default=original_config.seed)
parser.add_argument('-de',  '--destruction', type=float, default=original_config.destruction_quantity)
parser.add_argument('-gn', '--graph_name', type=str, default=original_config.graph_path)

# -----> END of variable parameters

config = None


def save_stats_monotonous(stats, fname):
    """ saving number of repairs and flow routed """
    for i in stats:
        print(i)

    repairs, iter, flow_cum = [], [], []
    n_repairs = 0
    demand_pairs = {k: [] for k in stats[-1]["demands_sat"].keys()}
    for i, dic in enumerate(stats):  # iteration index
        vals = dic["node"] + dic["edge"]
        # numbers in this iteration, to propagate values accordingly
        n_vals = max(len(vals), 1)
        repairs += vals if len(vals) > 0 else [None]
        iter += [dic["iter"]] * n_vals
        flow_cum += [stats[i]["flow"]] * n_vals
        n_repairs += len(vals)

        i_demand_pairs = stats[i]["demands_sat"] if "demands_sat" in stats[i].keys() else []
        for k in i_demand_pairs:
            d_flow = [0] * n_vals
            d_flow[0] = stats[i]["demands_sat"][k][i]
            demand_pairs[k] += d_flow

    df = pd.DataFrame()
    df["repairs"] = repairs
    df["iter"] = iter
    df["flow_cum"] = flow_cum

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


def setup_configuration():
    """ Sets up the configuration by assigning dynamic values to variables."""
    args = parser.parse_args()
    exec_config = configuration.Configuration()
    config_vars = [field for field in exec_config.__dict__]     # list of possible config fields

    for arg in vars(args):
        if arg in config_vars:
            setattr(exec_config, arg, getattr(args, arg))

    return exec_config


def print_configuration(config):
    """ Prints the configuration about to run as {key}\t{value}\n format. """
    str_values = ""
    for param, val in config.__dict__.items():
        str_values += "{}\t{}\n".format(param.upper(), val)
    str_values = str_values.strip()
    return str_values


def safe_run(*args):
    try:
        return run_single(*args)
    except Exception:
        enable_print()
        exec_details = fname_formation()
        util.write_file(exec_details + "\n", co.PATH_TO_FAILED_TESTS.format(time_batch_exec), is_append=True)
        print("error due to", exec_details)
        print(traceback.format_exc())
        print("stopped.")
        disable_print()


def fname_formation():
    global config
    fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-bud|{}-pbro|{}-idv|{}.csv".format(config.seed,
                                                                                config.graph_dataset.name,
                                                                                config.n_demand_pairs,
                                                                                int(config.demand_capacity),
                                                                                config.supply_capacity,
                                                                                config.algo_name.value[co.AlgoAttributes.NAME],
                                                                                config.monitors_budget,
                                                                                config.destruction_quantity,
                                                                                config.experiment_ind_var.value[0])
    return fname


def run_single(seed, dis, budget, nnodes, flowpp, indvar, algo_name, is_log=False):
    algo_name_o = co.Algorithm[algo_name]
    rep_mode = algo_name_o.value[co.AlgoAttributes.REPAIRING_PATH]
    pick_mode = algo_name_o.value[co.AlgoAttributes.PICKING_PATH]
    monitor_placement = algo_name_o.value[co.AlgoAttributes.MONITOR_PLACEMENT]
    monitoring_type = algo_name_o.value[co.AlgoAttributes.MONITORING_TYPE]

    __run_single(seed, dis, budget, nnodes, flowpp, rep_mode, pick_mode, monitor_placement, indvar, monitoring_type, algo_name, is_log)


def __run_single(seed, dis, budget, nnodes, flowpp, rep_mode, pick_mode, monitor_placement, indvar, monitoring_type, algo_name, is_log=False):
    global config
    config = setup_configuration()

    algo_name_o = co.Algorithm[algo_name]
    config.algo_name = algo_name_o

    config.experiment_ind_var = indvar
    config.is_log = is_log
    config.seed = seed
    config.destruction_quantity = dis

    # the prior probability that the node is broke is higher when the actual destruction is high
    if config.is_dynamic_prior:
        config.UNK_prior = config.destruction_quantity

    config.rand_generator_capacities = np.random.RandomState(config.seed)
    config.rand_generator_path_choice = np.random.RandomState(config.seed)

    config.protocol_monitor_placement = monitor_placement

    if config.algo_name == co.Algorithm.ORACLE:
        config.is_oracle_baseline = True

    if config.protocol_monitor_placement == co.ProtocolMonitorPlacement.STEP_BY_STEP_INFINITE:
        config.monitors_budget = np.inf  # this will be set to the max between budget and number of nodes
    else:
        config.monitors_budget = budget  # this will be set to the max between budget and number of nodes
    config.monitors_budget_residual = config.monitors_budget

    if config.is_demand_clique:
        config.n_demand_clique = nnodes
        config.n_demand_pairs = int(nnodes * (nnodes-1) / 2 * config.demand_clique_factor)
    else:
        config.n_demand_pairs = nnodes

    config.edges_list_path += str(config.n_demand_clique) + ".data"
    config.demand_capacity = flowpp
    config.repairing_mode = rep_mode
    config.picking_mode = pick_mode

    config.monitoring_type = monitoring_type
    config_details = print_configuration(config)

    fname = fname_formation()

    if config.force_recompute or not os.path.exists(co.PATH_EXPERIMENTS + fname):
        print()
        # print("NOW running...\n\n", config_details)
        print("exec name > ", fname)

        set_seeds(config.seed)

        if not config.is_log:
            disable_print()

        # RUNNING
        stats = config.algo_name.value[co.AlgoAttributes.EXEC].run(config)

        enable_print()

        if stats is not None:
            if config.algo_name in [co.Algorithm.SHP, co.Algorithm.ISR_MULTICOM,
                                    co.Algorithm.SHP_MONITOR, co.Algorithm.ISR_MULTICOM_MONITOR]:
                df = save_stats_NON_monotonous(stats, fname)
            else:
                df = save_stats_monotonous(stats, fname)
            print(df.to_string())
    else:
        print()
        print("THIS already existed...\n", fname, "\n")


def parallel_exec(seeds, algorithms, is_log=False):

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
                   #1: np.inf,
                   #2: np.inf,
                   3: [10, 12, 14, 16, 18, 20]
                   }

    ind_var = {0: [co.IndependentVariable.PROB_BROKEN, dis_uni],
               #1: [co.IndependentVariable.N_DEMAND_EDGES, npairs],
               #2: [co.IndependentVariable.FLOW_DEMAND, flowpp],
               3: [co.IndependentVariable.MONITOR_BUDGET, monitor_bud]
               }

    processes = []
    for seed in seeds:
        for k in ind_var:
            ind_variable_values = ind_var[k][1][k].copy()  # [list of x axis values]
            for iv in ind_variable_values:
                ind_var[k][1][k] = iv

                for algo_bench in algorithms:
                    exec_config = {
                        co.IndependentVariable.SEED: seed,
                        co.IndependentVariable.PROB_BROKEN: dis_uni[k],
                        co.IndependentVariable.MONITOR_BUDGET: monitor_bud[k],
                        co.IndependentVariable.N_DEMAND_EDGES: npairs[k],  # or nodes
                        co.IndependentVariable.FLOW_DEMAND: flowpp[k],
                        co.IndependentVariable.IND_VAR: ind_var[k][0],
                        co.IndependentVariable.ALGORITHM: algo_bench.name
                    }
                    processes.append(list(exec_config.values()) + [is_log])
            ind_var[k][1][k] = ind_variable_values  # reset

    with Pool(initializer=initializer, processes=co.N_CORES) as pool:
        try:
            pool.starmap(safe_run, processes)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()

    print("COMPLETED SUCCESSFULLY")


def single_exec():
    # BENCHMARKS = [co.Algorithm.TOMO_CEDAR, co.Algorithm.ORACLE, co.Algorithm.ST_PATH,
    #               co.Algorithm.CEDAR, co.Algorithm.SHP, co.Algorithm.ISR_SP, co.Algorithm.ISR_MULTICOM]

    BENCHMARKS = [co.Algorithm.TOMO_CEDAR_DYN]
    for algo in BENCHMARKS:
        exec_config = {
            co.IndependentVariable.SEED: 9124,
            co.IndependentVariable.PROB_BROKEN: 0.8,
            co.IndependentVariable.MONITOR_BUDGET: 7,
            co.IndependentVariable.N_DEMAND_EDGES: 5,  # nodes
            co.IndependentVariable.FLOW_DEMAND: 11,
            co.IndependentVariable.IND_VAR: co.IndependentVariable.PROB_BROKEN,
            co.IndependentVariable.ALGORITHM: algo.name
        }
        run_single(*exec_config.values(), True)


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


if __name__ == '__main__':

    # STEP = 3
    # s_seed, e_seed = 700, 800
    # seeds = set(range(s_seed, e_seed))
    # # seeds to ignore
    # seeds -= {700, 701, 703, 705, 714, 717, 721, 720, 722, 724, 726, 731, 736, 738, 740, 741, 744, 748, 758, 759, 752,
    #           760, 749, 761, 755, 783, 787, 794, 770, 765, 769, 774, 778, 782, 791, 792}
    # seeds = list(seeds)
    #
    # BENCHMARKS = [co.Algorithm.TOMO_CEDAR, co.Algorithm.ORACLE, co.Algorithm.CEDAR,
    #               co.Algorithm.ST_PATH, co.Algorithm.SHP, co.Algorithm.ISR_SP,
    #               co.Algorithm.ISR_MULTICOM]
    #
    # # BENCHMARKS = [co.Algorithm.SHP_MONITOR, co.Algorithm.CEDAR_MONITOR, co.Algorithm.TOMO_CEDAR_MONITOR,
    # #               co.Algorithm.ISR_MULTICOM_MONITOR, co.Algorithm.ISR_SP_MONITOR]
    #
    # # BENCHMARKS = [co.Algorithm.CEDAR_MONITOR]
    #
    # for i in range(0, len(seeds), STEP):
    #     runs = seeds[i: i+STEP]
    #     print("RUN", runs)
    #     parallel_exec(runs, BENCHMARKS, is_log=False)

    single_exec()
