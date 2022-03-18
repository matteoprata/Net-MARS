
import numpy as np
import itertools
import argparse
from multiprocessing import Pool

import signal
import src.constants as co
import src.configuration as configuration
import src.main_cedar_setup as main_cedar_setup
from src.utilities.util import set_seeds, block_print, enable_print

from src.plotting.stats_plotting import save_stats_as_df_ph1, plot_integral, plot_monitors_stuff

original_config = configuration.Configuration()

# -----> BEGIN of variable parameters

parser = argparse.ArgumentParser(description='Tomo Cedar recovery algorithm run parameters.')
parser.add_argument('-s',  '--seed', type=int, default=original_config.seed)
parser.add_argument('-de',  '--destruction', type=float, default=original_config.destruction_quantity)
parser.add_argument('-gn', '--graph_name', type=str, default=original_config.graph_path)

# -----> END of variable parameters


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


# def execution_name(config):
#     return "s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-pbro|{}".format(config.seed, config.graph_path, config.n_demand_pairs,
#                                                                 config.demand_capacity, config.supply_capacity, config.algo_name, config.destruction_quantity)


def run_var_seed_dis(seed, dis, budget, nnodes, flowpp, rep_mode, pick_mode, is_parallel=False):
    config = setup_configuration()

    config.mute_log = is_parallel
    config.seed = seed
    config.destruction_quantity = dis

    # the prior probability that the node is broke is higher when the actual destruction is high
    if config.is_dynamic_prior:
        config.UNK_prior = config.destruction_quantity

    config.rand_generator_capacities = np.random.RandomState(config.seed)
    config.rand_generator_path_choice = np.random.RandomState(config.seed)
    config.monitors_budget = budget
    config.n_demand_clique = nnodes
    config.demand_capacity = flowpp
    config.repairing_mode = rep_mode
    config.picking_mode = pick_mode

    config_details = print_configuration(config)
    print()
    print("NOW running...\n\n", config_details, "\n")

    set_seeds(config.seed)

    if config.mute_log:
        block_print()

    stats = main_cedar_setup.run(config)
    enable_print()
    if stats is not None:
        save_stats_as_df_ph1(stats, config)


def parallel_exec():

    # seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # dis_uni = [.3]   # [.05, .15, .3, .5, .7]
    # budget_n_monitor = [20]   # [10, 15, 20, 25, 30, 40, 50]
    # npairs = [5, 6, 7, 8, 9, 10]     # [5, 6, 7, 8, 9, 10]
    # flowpp = [10.0]  # [5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
    # reps = [co.ProtocolRepairingPath.MIN_COST_BOT_CAP,
    #             co.ProtocolRepairingPath.SHORTEST,
    #             co.ProtocolRepairingPath.MAX_BOT_CAP,
    #             co.ProtocolRepairingPath.IP]
    #
    # pick = [co.ProtocolPickingPath.RANDOM,
    #         co.ProtocolPickingPath.MIN_COST_BOT_CAP,
    #         co.ProtocolPickingPath.MAX_BOT_CAP]

    seeds = range(30, 39)
    dis_uni = [.05, .15, .3, .5, .7]
    budget_n_monitor = [20]
    npairs = [8]
    flowpp = [10.0]

    reps = [co.ProtocolRepairingPath.SHORTEST,
            co.ProtocolRepairingPath.MAX_BOT_CAP,
            co.ProtocolRepairingPath.MIN_COST_BOT_CAP,
            ]  # co.ProtocolRepairingPath.IP

    pick = [co.ProtocolPickingPath.RANDOM,
            co.ProtocolPickingPath.MIN_COST_BOT_CAP,
            co.ProtocolPickingPath.MAX_BOT_CAP]

    processes = []
    for execution in itertools.product(seeds, dis_uni, budget_n_monitor, npairs, flowpp, reps, pick, [True]):
        processes.append(execution)

    with Pool(initializer=initializer, processes=co.N_CORES) as pool:
        try:
            pool.starmap(run_var_seed_dis, processes)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()

    print("COMPLETED SUCCESSFULLY")


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def plotting_data():
    config = setup_configuration()
    seeds = list(range(1, 10)) + list(range(11, 19)) + list(range(30, 39))
    dis_uni = [.05, .15, .3, .5, .7]

    algos = [("CEDARNEW_BUD_20", "{}I{}".format(i,j)) for i in range(3) for j in range(3)]
    # algos += [("CEDARNEW_BUD_20", "{}I{}".format(i,j)) for i in range(1) for j in range(2,3)]

    # algos += [("CEDARNEW_BUD_20", "{}I{}".format(2, 2))]
    # algos += [("CEDARNEW_BUD_20", "{}I{}".format(2, 1))]

    source = "data/experiments/"

    plot_integral(source, config, seeds, dis_uni, algos, is_total=False, x_position=0)
    plot_integral(source, config, seeds, dis_uni, algos, is_total=True, x_position=0)
    #
    plot_monitors_stuff(source, config, seeds, dis_uni, algos, typep="n_monitor_msg", x_position=0)
    plot_monitors_stuff(source, config, seeds, dis_uni, algos, typep="n_monitors", x_position=0)
    plot_monitors_stuff(source, config, seeds, dis_uni, algos, typep="n_repairs", x_position=0)


if __name__ == '__main__':

    # parallel_exec()
    # plotting_data()
    run_var_seed_dis(seed=80, dis=.8, budget=20, nnodes=9, flowpp=10,
                      rep_mode=co.ProtocolRepairingPath.MAX_BOT_CAP,
                      pick_mode=co.ProtocolPickingPath.MIN_COST_BOT_CAP)
