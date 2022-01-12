
import numpy as np
import random
import argparse
from multiprocessing import Pool

import src.constants as co
import src.configuration as configuration
import src.main_cedar_setup as main_cedar_setup
from utilities.util import set_seeds, block_print, enable_print

from src.plotting.stats_plotting import save_stats_as_df_ph1, plot_integral, plot_monitors_stuff

original_config = configuration.Configuration()

# -----> BEGIN of variable parameters

parser = argparse.ArgumentParser(description='Tomo Cedar recovery algorithm run parameters.')
parser.add_argument('-s',  '--seed', type=int, default=original_config.seed)
parser.add_argument('-de',  '--destruction', type=float, default=original_config.destruction_uniform_quantity)
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


def execution_name(config):
    return "s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-pbro|{}".format(config.seed, config.graph_path, config.n_demand_pairs,
                                                                config.demand_capacity, config.supply_capacity, config.algo_name, config.destruction_uniform_quantity)


def run_var_seed_dis(seed, dis):
    config = setup_configuration()

    config.seed = seed
    config.destruction_uniform_quantity = dis
    config.rand_generator_capacities = np.random.RandomState(config.seed)

    config_details = print_configuration(config)
    print()
    print("NOW running...\n\n", config_details, "\n")

    set_seeds(config.seed)

    block_print()
    stats = main_cedar_setup.run(config)
    enable_print()

    save_stats_as_df_ph1(stats, config)


def exec():
    seeds = [4, 5, 6]  # , 4, 5, 6]  # , 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # range(5), 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    dis_uni = [.05, .15, .3, .5, .7]  # .05, .15, .3, .5, .7]  # [.05, .15, .30]

    processes = []
    for seed in seeds:
        for dis in dis_uni:
            processes.append((seed, dis))

    with Pool(processes=co.N_CORES) as pool:
        pool.starmap(run_var_seed_dis, processes)


def plotting_data():
    config = setup_configuration()
    seeds = [5, 9, 10, 15, 17]  # [1, 3, 5, 9, 10, 15, 17, 19]
    dis_uni = [.05, .15, .3, .5, .7]
    algos = ["CEDAR", "CEDARNEW"]
    source = "data/experiments/"

    plot_integral(source, config, seeds, dis_uni, algos)
    plot_monitors_stuff(source, config, seeds, dis_uni, algos, typep="n_monitor_msg")
    plot_monitors_stuff(source, config, seeds, dis_uni, algos, typep="n_monitors")


if __name__ == '__main__':
    exec()
    #plotting_data()


