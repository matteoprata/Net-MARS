
import numpy as np
import random
import argparse

import src.constants as co
import src.configuration as configuration
import src.main_cedar_setup as main_cedar_setup

from src.plotting.stats_plotting import save_stats_as_df_ph1, aggregate_multiple_seeds

original_config = configuration.Configuration()

# -----> BEGIN of variable parameters

parser = argparse.ArgumentParser(description='Tomo Cedar recovery algorithm run parameters.')
parser.add_argument('-s', '--seed', type=int, default=original_config.seed)
parser.add_argument('-gn', '--graph_name', type=str, default=original_config.graph_path)
parser.add_argument('-de', '--destruction', type=str, default=original_config.destruction_type.value)

# -----> END of variable parameters


def setup_configuration():
    """ Sets up the configuration by assigning dynamic values to variables."""
    args = parser.parse_args()
    exec_config = configuration.Configuration()
    config_vars = [field for field in exec_config.__dict__]     # list of possible config fields

    for arg in vars(args):
        if arg in config_vars:
            setattr(exec_config, arg, getattr(args, arg))

    exec_config.rand_generator_capacities = np.random.RandomState(exec_config.seed)
    return exec_config


def print_configuration(config):
    """ Prints the configuration about to run as {key}\t{value}\n format. """
    str_values = ""
    for param, val in config.__dict__.items():
        str_values += "{}\t{}\n".format(param.upper(), val)
    str_values = str_values.strip()
    return str_values


def execution_name(config):
    return "s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}".format(config.seed, config.graph_path, config.n_demand_pairs,
                                                        config.demand_capacity, config.supply_capacity, config.algo_name)


def run_tomo_cedar(config):
    """ Run the program. """

    np.random.seed(config.seed)
    random.seed(config.seed)
    stats = main_cedar_setup.run(config)
    save_stats_as_df_ph1(stats, config)


def exec(exec_config):
    seeds = range(13)
    # dem_pairs = [5]   # range(1, 8)
    # dem_flows = [15]  # range(5, 26, 5)

    # for n_pairs in dem_pairs:
    #     for scp in dem_flows:
    for s in seeds:

        # exec_config.demand_capacity = scp
        # exec_config.n_demand_pairs = n_pairs
        exec_config.seed = s

        # if exec_config.algo_name == co.AlgoName.TOMO_CEDAR.value:
        run_tomo_cedar(exec_config)


if __name__ == '__main__':
    exec_config = setup_configuration()
    config_details = print_configuration(exec_config)

    print("NOW running...\n\n", config_details, "\n")
    exec(exec_config)
    aggregate_multiple_seeds(exec_config)



