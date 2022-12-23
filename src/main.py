
import numpy as np
import argparse
from multiprocessing import Pool
import os

import traceback
import signal
import src.constants as co
import src.configuration as configuration

from src.utilities.store_recovery_stats import save_stats_monotonous, save_stats_NON_monotonous
from src.utilities.util import set_seed, disable_print, enable_print
import src.utilities.util as util
import time

original_config = configuration.Configuration()
time_batch_exec = time.strftime("%Y-%m-%d_%H-%M")


config = None


def cli_args():
    """
    Parse the CLI arguments.
    :return: parsed cli arguments
    """

    # -----> BEGIN of variable parameters

    parser = argparse.ArgumentParser(description='NetMARS parameters.')
    parser.add_argument('-gn', '--graph_name', type=str, default=original_config.graph_path)
    parser.add_argument('-par', '--is_parallel', type=int, default=original_config.is_parallel)
    parser.add_argument('-set', '--setup', type=str)

    # -----> END of variable parameters

    parsed_arguments = parser.parse_args()
    return parsed_arguments


parsed_arguments = cli_args()  # reads CLI arguments


def setup_configuration():
    """ Sets up the configuration by assigning dynamic values to variables."""
    global parsed_arguments
    exec_config = configuration.Configuration()
    config_vars = [field for field in exec_config.__dict__]     # list of possible config fields

    for arg in vars(parsed_arguments):
        if arg in config_vars:
            setattr(exec_config, arg, getattr(parsed_arguments, arg))

    return exec_config


def print_configuration(config):
    """ Prints the configuration about to run as {key}\t{value}\n format. """
    str_values = ""
    for param, val in config.__dict__.items():
        str_values += "{}\t{}\n".format(param.upper(), val)
    str_values = str_values.strip()
    return str_values


def safe_run(*args):
    global config

    try:
        return run_single(*args)

    except Exception:
        enable_print()
        print("QUA")
        exec_details = fname_formation()
        trace = traceback.format_exc()
        util.write_file(exec_details + "\n" + trace + "\n\n", co.PATH_TO_FAILED_TESTS.format(time_batch_exec), is_append=True)

        # writes the infeasible seed only once
        if os.path.exists(co.PATH_TO_FAILED_SEEDS):
            fs = util.read_file(co.PATH_TO_FAILED_SEEDS)
            if not str(config.seed) in fs:
                util.write_file(str(config.seed) + "\n", co.PATH_TO_FAILED_SEEDS, is_append=True)
        else:
            util.write_file(str(config.seed) + "\n", co.PATH_TO_FAILED_SEEDS, is_append=True)

        print("error due to", exec_details)
        print(trace)
        print("printed traceback but ignored.")
        disable_print()


def fname_formation():
    global config
    fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-bud|{}-pbro|{}-idv|{}.csv".format(config.seed,
                                                                                       config.graph_dataset.name,
                                                                                       config.n_edges_demand,
                                                                                       int(config.demand_capacity),
                                                                                       config.supply_capacity,
                                                                                       config.algo_name.value[co.AlgoAttributes.FILE_NAME],
                                                                                       config.monitors_budget,
                                                                                       config.destruction_quantity,
                                                                                       config.experiment_ind_var.value[0])
    return fname


def run_single(algo_name, seed, dis, budget, n_dedges, flowpp, indvar, is_log=False):
    algo_name_o = co.Algorithm[algo_name]
    rep_mode = algo_name_o.value[co.AlgoAttributes.REPAIRING_PATH]
    pick_mode = algo_name_o.value[co.AlgoAttributes.PICKING_PATH]
    monitor_placement = algo_name_o.value[co.AlgoAttributes.MONITOR_PLACEMENT]
    monitoring_type = algo_name_o.value[co.AlgoAttributes.MONITORING_TYPE]

    global config

    config = setup_configuration()  # MUST BE ON TOP
    config.seed = seed

    algo_name_o = co.Algorithm[algo_name]
    config.algo_name = algo_name_o

    config.experiment_ind_var = indvar
    config.is_log = is_log
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
        config.n_nodes_demand_clique = len(co.FIXED_DEMAND_NODES)  # is CONSTANT
        assert n_dedges <= config.n_nodes_demand_clique * (config.n_nodes_demand_clique - 1) / 2, "add more more demand nodes in co.FIXED_DEMAND_NODES to work with {} demand edges.".format(n_dedges)
        config.n_edges_demand = n_dedges  # int(nnodes * (nnodes-1) / 2 * config.demand_clique_factor)
    else:
        config.n_edges_demand = n_dedges

    config.edges_list_path += str(config.n_nodes_demand_clique) + ".data"
    config.demand_capacity = flowpp
    config.repairing_mode = rep_mode
    config.picking_mode = pick_mode

    config.monitoring_type = monitoring_type
    config_details = print_configuration(config)

    fname = fname_formation()

    # check if seed is ok
    # if os.path.exists(co.PATH_TO_FAILED_SEEDS):
    #     if config.is_cluster_execution:
    #         fs = set(util.read_file(co.PATH_TO_FAILED_SEEDS))
    #         if str(config.seed) in fs:
    #             raise Exception()
    #     else:
    #         warnings.warn("CAREFUL! Running local, but this seed is marked as BAD. Check {}".format(co.PATH_TO_FAILED_SEEDS))

    if config.force_recompute or not os.path.exists(co.PATH_EXPERIMENTS + fname):
        print()
        # print("NOW running...\n\n", config_details)
        print("exec name > ", fname)

        set_seed(config.seed)

        if not config.is_log:
            disable_print()

        # RUNNING
        stats = config.algo_name.value[co.AlgoAttributes.EXEC].run(config)

        enable_print()

        if stats is not None:
            if config.algo_name in [co.Algorithm.SHP, co.Algorithm.ISR_MULTICOM]:
                df = save_stats_NON_monotonous(stats, fname)
            else:
                df = save_stats_monotonous(stats, fname, config.algo_name)
            print(df.to_string())
    else:
        print()
        print("THIS already existed...\n", fname, "\n")


def main(setup, is_parallel):
    """
    Runs the simulation according to a setup in src.experimental_setup.
    :param setup: a setup file in src.experimental_setup.
    :param is_parallel: weather to run the simulation in parallel or not.
    :return:
    """

    # 1. Declare independent variables and their domain
    # 2. Declare what independent variable varies at this execution and what stays fixed

    processes = []
    indv_fixed_original = {k: setup.indv_fixed[k] for k in setup.indv_fixed}
    for a in setup.comparison_dims[co.IndependentVariable.ALGORITHM]:
        for s in setup.comparison_dims[co.IndependentVariable.SEED]:

            for x_var_k in setup.indv_vary:  # execute for several independent variables
                X_var = setup.indv_vary[x_var_k]
                for x in X_var:  # iterates over the ind var values
                    setup.indv_fixed[x_var_k] = x

                    # declare processes arguments
                    process = [a.name, s] + list(setup.indv_fixed.values()) + [x_var_k, is_parallel]
                    processes.append(process)
                    setup.indv_fixed = {k: indv_fixed_original[k] for k in indv_fixed_original}  # reset the change

    if is_parallel:
        print("Running parallely...")
        execute_parallel_processes(safe_run, processes)
    else:
        print("Running sequentially...")
        for p in processes:
            safe_run(*p)


def execute_parallel_processes(func_exe, func_args: list):
    """
    Runs processes in parallel. Given the function to run and its arguments.
    :param func_exe: function to run.
    :param func_args: arguments.
    """
    initializer = signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ignore CTRL+C in the worker process.
    with Pool(initializer=initializer, processes=co.N_CORES) as pool:
        try:
            pool.starmap(func_exe, func_args)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()

    print("COMPLETED SUCCESSFULLY")


def get_setup_file(chosen_setup):
    from src.experimental_setup import setup_01, setup_02

    setups = {
        "setup_01": setup_01,
        "setup_02": setup_02
    }
    if chosen_setup in setups.keys():
        return setups[chosen_setup]
    else:
        print("No setup imported named {}.".format(chosen_setup))
        exit()


if __name__ == '__main__':
    setup_file = get_setup_file(parsed_arguments.setup)  # interprets CLI arguments
    main(setup_file, bool(parsed_arguments.is_parallel))
