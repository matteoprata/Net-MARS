
import numpy as np
import argparse
import os

import traceback
import src.constants as co
import src.configuration as configuration

from src.utilities.store_recovery_stats import save_stats_monotonous, save_stats_NON_monotonous
from src.utilities.util import set_seed, disable_print, enable_print
from src.recovery_protocols import RecoveryProtocol
import src.utilities.util as util
import time


def cli_args_parsing():
    """
    Parse the CLI arguments.
    :return: parsed cli arguments
    """

    parser = argparse.ArgumentParser(description='NetMARS parameters.')

    # -----> BEGIN of variable parameters
    parser.add_argument('-set', '--setup', type=str)
    parser.add_argument('-par', '--is_parallel', type=int, default=False)
    parser.add_argument('-log', '--is_log', type=int, default=True)
    # -----> END of variable parameters

    parsed_arguments = vars(parser.parse_args())

    # processing args
    parsed_arguments["setup"] = co.Setups[parsed_arguments["setup"].upper()].value
    parsed_arguments["is_parallel"] = bool(parsed_arguments["is_parallel"])
    parsed_arguments["is_log"] = bool(parsed_arguments["is_log"])

    return parsed_arguments


def configuration_update(config):
    """ Sets up the configuration by assigning dynamic values to variables of configuration.
        Notice: CLI arguments must have the same name as those in the configuration.py file.
    """
    global parsed_arguments
    config_vars = [field for field in config.__dict__]     # list of possible config fields

    for arg in parsed_arguments:
        if arg in config_vars:
            setattr(config, arg, parsed_arguments[arg])
        else:
            print("WARNING! A CLI argument is not in the configuration file.")
    return config


def configuration_details(config):
    """ Prints the configuration running. As {key}\t{value}\n format. """
    str_values = ""
    for param, val in config.__dict__.items():
        str_values += "{}\t{}\n".format(param.upper(), val)
    str_values = str_values.strip()
    return str_values


def safe_run(*args):

    config = run_0_set_configuration_values(*args)
    try:
        return run_1_single_execution(config)

    except Exception:

        enable_print()
        exec_details = fname_formation(config)
        trace = traceback.format_exc()
        util.write_file(exec_details + "\n" + trace + "\n\n", co.PATH_TO_FAILED_TESTS.format(time_batch_exec), is_append=True)

        # writes the infeasible seeds in a file, only once
        if os.path.exists(co.PATH_TO_FAILED_SEEDS):
            fs = util.read_file(co.PATH_TO_FAILED_SEEDS)
            if not str(config.seed) in fs:
                util.write_file(str(config.seed) + "\n", co.PATH_TO_FAILED_SEEDS, is_append=True)
        else:
            util.write_file(str(config.seed) + "\n", co.PATH_TO_FAILED_SEEDS, is_append=True)

        print("error due to", exec_details)
        print(trace)
        disable_print()


def fname_formation(config):
    fname = "seed={}-g={}-np={}-dc={}-spc={}-alg={}-bud={}-pbro={}-idv={}.csv".format(config.seed,
                                                                                       config.graph_dataset.name,
                                                                                       config.n_edges_demand,
                                                                                       int(config.demand_capacity),
                                                                                       config.supply_capacity,
                                                                                       config.algo_instance.file_name,
                                                                                       config.monitors_budget,
                                                                                       config.destruction_quantity,
                                                                                       config.experiment_ind_var.value[0])
    return fname


def run_0_set_configuration_values(graph_dataset, algo_name, seed, dis, budget, n_dedges, flowpp, indvar):
    """
    Sets the desired parameters in a configuration file available throughout the simulation.
    :return: a configuration object.
    """

    # create a config file
    config = configuration.Configuration()
    config = configuration_update(config)

    repairing_protocol: RecoveryProtocol = co.Algorithm[algo_name].value(config)

    config.seed = seed
    config.algo_name = co.Algorithm[algo_name]
    config.algo_instance = repairing_protocol
    config.graph_dataset = graph_dataset
    config.graph_path = graph_dataset.value
    config.experiment_ind_var = indvar
    config.destruction_quantity = dis

    # the prior probability that the node is broke is higher when the actual destruction is high
    if config.is_dynamic_prior:
        config.UNK_prior = config.destruction_quantity

    config.rand_generator_capacities = np.random.RandomState(config.seed)
    config.rand_generator_path_choice = np.random.RandomState(config.seed)

    config.protocol_monitor_placement = repairing_protocol.mode_monitoring

    if config.algo_instance == co.Algorithm.PROTON_ORACLE:
        config.is_oracle_baseline = True

    if config.protocol_monitor_placement == co.ProtocolMonitorPlacement.STEP_BY_STEP_INFINITE:
        config.monitors_budget = np.inf  # this will be set to the max between budget and number of nodes
    else:
        config.monitors_budget = budget  # this will be set to the max between budget and number of nodes
    config.monitors_budget_residual = config.monitors_budget

    if config.is_demand_clique:
        config.n_nodes_demand_clique = len(co.FIXED_DEMAND_NODES)  # is CONSTANT
        assert_msg = "add more more demand nodes in co.FIXED_DEMAND_NODES to work with {} demand edges.".format(n_dedges)
        assert n_dedges <= config.n_nodes_demand_clique * (config.n_nodes_demand_clique - 1) / 2, assert_msg
        config.n_edges_demand = n_dedges  # int(nnodes * (nnodes-1) / 2 * config.demand_clique_factor)
    else:
        config.n_edges_demand = n_dedges

    config.edges_list_path += str(config.n_nodes_demand_clique) + ".data"
    config.demand_capacity = flowpp
    config.repairing_mode = repairing_protocol.mode_path_repairing
    config.picking_mode = repairing_protocol.mode_path_choosing_repair

    config.monitoring_type = repairing_protocol.mode_monitoring_type
    return config


def run_1_single_execution(config):
    """ Executes the simulation. """

    confid_description = configuration_details(config)
    fname = fname_formation(config)

    if not (config.force_recompute or not os.path.exists(co.PATH_EXPERIMENTS + fname)):
        print()
        print("THIS already existed...\n", fname, "\n")
        return

    print("\nNOW running...\n\n{}\n".format(confid_description))
    print("\nexec fname > ", fname)

    set_seed(config.seed)

    if not config.is_log:
        disable_print()

    # RUNNING
    stats = config.algo_instance.run()

    enable_print()

    if stats is not None:
        if config.algo_instance in [co.Algorithm.SHP, co.Algorithm.ISR_MULTICOM]:
            df = save_stats_NON_monotonous(stats, fname)
        else:
            df = save_stats_monotonous(stats, fname, config.algo_name)
        print(df.to_string())


def main(setup, is_parallel):
    """
    Runs the simulation according to a setup in src.experimental_setup.
    :param setup: a setup file in src.experimental_setup.
    :param is_parallel: weather to run the simulation in parallel or not.
    :return:
    """

    # 1. Declare independent variables and their domain in a setup file at src.experimental_setup
    # 2. Declare what independent variable varies at this execution and what stays fixed in the file

    processes = []
    indv_fixed_original = {k: setup.indv_fixed[k] for k in setup.indv_fixed}

    for s in setup.comparison_dims[co.IndependentVariable.SEED]:
        for g in setup.comparison_dims[co.IndependentVariable.GRAPH]:
            for a in setup.comparison_dims[co.IndependentVariable.ALGORITHM]:

                for x_var_k in setup.indv_vary:  # execute for several independent variables
                    X_var = setup.indv_vary[x_var_k]
                    for x in X_var:  # iterates over the ind var values
                        setup.indv_fixed[x_var_k] = x

                        # declare processes arguments
                        process = [g, a.name, s] + list(setup.indv_fixed.values()) + [x_var_k]
                        processes.append(process)
                        setup.indv_fixed = {k: indv_fixed_original[k] for k in indv_fixed_original}  # reset the change

    # RUNNING
    if is_parallel:
        print("Running parallely...")
        util.execute_parallel_processes(safe_run, processes, co.N_CORES)
    else:
        print("Running sequentially...")
        for p in processes:
            safe_run(*p)


# --- (begin) PREAMBLE --- #

time_batch_exec = time.strftime("%Y-%model-%d_%H-%M")
parsed_arguments = cli_args_parsing()

# --- (end) PREAMBLE --- #


if __name__ == '__main__':
    setup, is_parallel = parsed_arguments["setup"], parsed_arguments["is_parallel"]
    main(setup, is_parallel)
