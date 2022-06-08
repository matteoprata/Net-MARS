
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
import src.recovery_protocols.main_stocedar_setup as main_stocedar_setup
import src.recovery_protocols.main_cedar_setup as main_cedar_setup
import src.recovery_protocols.main_shp_setup as main_shp_setup

from src.utilities.util import set_seeds, disable_print, enable_print

original_config = configuration.Configuration()

# -----> BEGIN of variable parameters

parser = argparse.ArgumentParser(description='Tomo Cedar recovery algorithm run parameters.')
parser.add_argument('-s',  '--seed', type=int, default=original_config.seed)
parser.add_argument('-de',  '--destruction', type=float, default=original_config.destruction_quantity)
parser.add_argument('-gn', '--graph_name', type=str, default=original_config.graph_path)

# -----> END of variable parameters

config = None


def save_stats_as_df_ph1(stats, fname):
    """ saving number of repairs and flow routed """
    for i in stats:
        print(i)

    repairs, iter, flow_cum = [], [], []
    n_repairs = 0
    for i, dic in enumerate(stats):
        vals = dic["node"] + dic["edge"]
        n_vals = max(len(vals), 1)
        repairs += vals if len(vals) > 0 else [None]
        iter += [dic["iter"]] * n_vals
        flow_cum += [stats[i]["flow"]] * n_vals
        n_repairs += len(vals)

    # flow_cum[-1] = stats[-1]["flow"]

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

    print("saving stats > {}".format(fname))
    df.to_csv("data/experiments/{}".format(fname))
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
        return run_var_seed_dis(*args)
    except Exception:
        enable_print()
        print(traceback.format_exc())
        print("due to", fname_formation())
        disable_print()


def fname_formation():
    global config
    fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-bud|{}-pbro|{}-rep|{}-pik|{}-mop|{}.csv".format(config.seed,
                                                                                                     config.graph_dataset.name,
                                                                                                     config.n_demand_pairs,
                                                                                                     int(config.demand_capacity),
                                                                                                     config.supply_capacity,
                                                                                                     config.algo_name.value,
                                                                                                     config.monitors_budget,
                                                                                                     config.destruction_quantity,
                                                                                                     config.repairing_mode.value,
                                                                                                     config.picking_mode.value,
                                                                                                     config.protocol_monitor_placement.value)
    return fname


def run_var_seed_dis(seed, dis, budget, nnodes, flowpp, rep_mode, pick_mode, monitor_placement, indvar, monitoring_type, algo_name, is_parallel=False):
    global config
    config = setup_configuration()

    config.algo_name = algo_name
    config.experiment_ind_var = indvar
    config.mute_log = is_parallel
    config.seed = seed
    config.destruction_quantity = dis

    # the prior probability that the node is broke is higher when the actual destruction is high
    if config.is_dynamic_prior:
        config.UNK_prior = config.destruction_quantity

    config.rand_generator_capacities = np.random.RandomState(config.seed)
    config.rand_generator_path_choice = np.random.RandomState(config.seed)

    config.monitors_budget = budget  # this will be set to the max between budget and number of nodes
    config.monitors_budget_residual = config.monitors_budget

    if config.is_demand_clique:
        config.n_demand_clique = nnodes
        config.n_demand_pairs = int(nnodes * (nnodes-1) / 2 * config.demand_clique_factor)
    else:
        config.n_demand_pairs = nnodes

    config.demand_capacity = flowpp
    config.repairing_mode = rep_mode
    config.picking_mode = pick_mode

    config.protocol_monitor_placement = monitor_placement
    if config.protocol_monitor_placement == co.ProtocolMonitorPlacement.ORACLE:
        config.is_oracle_baseline = True

    config.monitoring_type = monitoring_type
    config_details = print_configuration(config)

    fname = fname_formation()

    if config.force_recompute or not os.path.exists("data/experiments/" + fname):
        print()
        # print("NOW running...\n\n", config_details)
        print("exec name > ", fname)

        set_seeds(config.seed)

        if config.mute_log:
            disable_print()

        if config.algo_name in [co.AlgoName.ISR, co.AlgoName.ISR_MULTICOM]:
            stats = main_stocedar_setup.run(config)
        elif config.algo_name == co.AlgoName.CEDAR:
            stats = main_cedar_setup.run(config)
        elif config.algo_name == co.AlgoName.SHP:
            stats = main_shp_setup.run(config)
        else:
            stats = main_tomocedar_setup.run(config)
        enable_print()

        if stats is not None:
            df = save_stats_as_df_ph1(stats, fname)
            print(df)
    else:
        print()
        print("THIS already existed...\n", fname, "\n")

    return


def parallel_exec(seeds):

    ind_var = {0: [co.IndependentVariable.PROB_BROKEN],
               1: [co.IndependentVariable.N_DEMAND_EDGES],
               2: [co.IndependentVariable.FLOW_DEMAND],
               3: [co.IndependentVariable.MONITOR_BUDGET]
               }

    dis_uni = {0: [.3, .4, .5, .6, .7, .8, .9],
               1: [.5],
               2: [.5],
               3: [.5]}

    npairs = {0: [8],
              1: [5, 6, 7, 8, 9, 10],
              2: [8],
              3: [8]}

    flowpp = {0: [11],
              1: [11],
              2: [5, 7, 9, 11, 13, 15],
              3: [11]}

    monitor_bud = {0: [25],
                   1: [25],
                   2: [25],
                   3: [15, 20, 25, 30, 35, 40]}

    processes = []
    for k in range(len(ind_var)):

        # SHORTEST
        for execution in itertools.product(seeds, dis_uni[k], monitor_bud[k], npairs[k], flowpp[k],
                                           [co.ProtocolRepairingPath.SHORTEST_MINUS],
                                           [co.ProtocolPickingPath.RANDOM],
                                           [co.ProtocolMonitorPlacement.NONE], ind_var[k],
                                           [co.PriorKnowledge.DUNNY_IP], [co.AlgoName.CEDARNEW], [True]):
            processes.append(execution)

        for execution in itertools.product(seeds, dis_uni[k], monitor_bud[k], npairs[k], flowpp[k],
                                           [co.ProtocolRepairingPath.MIN_COST_BOT_CAP],
                                           [co.ProtocolPickingPath.MIN_COST_BOT_CAP],
                                           [co.ProtocolMonitorPlacement.ORACLE,
                                            co.ProtocolMonitorPlacement.BUDGET], ind_var[k],
                                           [co.PriorKnowledge.TOMOGRAPHY], [co.AlgoName.CEDARNEW], [True]):
            processes.append(execution)

        for execution in itertools.product(seeds, dis_uni[k], monitor_bud[k], npairs[k], flowpp[k],
                                           [co.ProtocolRepairingPath.SHORTEST_MINUS],
                                           [co.ProtocolPickingPath.RANDOM],
                                           [co.ProtocolMonitorPlacement.NONE], ind_var[k],
                                           [co.PriorKnowledge.DUNNY_IP], [co.AlgoName.CEDAR], [True]):
            processes.append(execution)

        for execution in itertools.product(seeds, dis_uni[k], monitor_bud[k], npairs[k], flowpp[k],
                                           [co.ProtocolRepairingPath.SHORTEST_MINUS],
                                           [co.ProtocolPickingPath.RANDOM],
                                           [co.ProtocolMonitorPlacement.NONE], ind_var[k],
                                           [co.PriorKnowledge.DUNNY_IP], [co.AlgoName.ISR], [True]):
            processes.append(execution)

        for execution in itertools.product(seeds, dis_uni[k], monitor_bud[k], npairs[k], flowpp[k],
                                           [co.ProtocolRepairingPath.SHORTEST_MINUS],
                                           [co.ProtocolPickingPath.RANDOM],
                                           [co.ProtocolMonitorPlacement.NONE], ind_var[k],
                                           [co.PriorKnowledge.DUNNY_IP], [co.AlgoName.ISR_MULTICOM], [True]):
            processes.append(execution)

        # for execution in itertools.product(seeds, dis_uni[k], monitor_bud[k], npairs[k], flowpp[k],
        #                                    [co.ProtocolRepairingPath.SHORTEST_MINUS],
        #                                    [co.ProtocolPickingPath.RANDOM],
        #                                    [co.ProtocolMonitorPlacement.NONE], ind_var[k],
        #                                    [co.PriorKnowledge.DUNNY_IP], [co.AlgoName.SHP], [True]):
        #     processes.append(execution)

    with Pool(initializer=initializer, processes=co.N_CORES) as pool:
        try:
            pool.starmap(safe_run, processes)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()

    print("COMPLETED SUCCESSFULLY")


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


if __name__ == '__main__':
    # parallel_exec(seeds=range(50, 55))
    # print("DONE R1")
    # parallel_exec(seeds=range(0, 5))
    # print("DONE R2")
    # parallel_exec(seeds=range(100, 105))
    # print("DONE R3")
    # parallel_exec(seeds=range(550, 555))
    # print("DONE R4")

    # run_var_seed_dis(seed=100, dis=.6, budget=25, nnodes=11, flowpp=10,
    #                  rep_mode=co.ProtocolRepairingPath.SHORTEST_MINUS,
    #                  pick_mode=co.ProtocolPickingPath.RANDOM,
    #                  indvar=co.IndependentVariable.FLOW_DEMAND,
    #                  monitor_placement=co.ProtocolMonitorPlacement.NONE,
    #                  monitoring_type=co.PriorKnowledge.DUNNY_IP
    #                  )

    run_var_seed_dis(seed=50, dis=.3, budget=25, nnodes=5, flowpp=11,
                     rep_mode=co.ProtocolRepairingPath.MIN_COST_BOT_CAP,
                     pick_mode=co.ProtocolPickingPath.MIN_COST_BOT_CAP,
                     indvar=co.IndependentVariable.FLOW_DEMAND,
                     monitor_placement=co.ProtocolMonitorPlacement.BUDGET,
                     monitoring_type=co.PriorKnowledge.TOMOGRAPHY,

                     # KEY PARAM
                     algo_name=co.AlgoName.CEDARNEW
                     )
