
import numpy as np
import random
import argparse

import src.configuration as configuration
import src.main_setup as main_setup

original_config = configuration.Configuration()

# -----> BEGIN of variable parameters

parser = argparse.ArgumentParser(description='Tomo Cedar recovery algorithm run parameters.')
parser.add_argument('-s', '--seed', type=int, default=original_config.seed)
parser.add_argument('-gn', '--graph_name', type=str, default=original_config.graph_name)
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
    return exec_config


def print_configuration(config):
    """ Prints the configuration about to run as {key}\t{value}\n format. """
    str_values = ""
    for param, val in config.__dict__.items():
        str_values += "{}\t{}\n".format(param.upper(), val)
    str_values = str_values.strip()
    return str_values


def run_tomo_cedar(config):
    """ Run the program. """
    np.random.seed(config.seed)
    random.seed(config.seed)
    main_setup.run(config)


if __name__ == '__main__':
    exec_config = setup_configuration()
    config_details = print_configuration(exec_config)

    print("NOW running...\n\n", config_details, "\n")

    # seeds = range(1)
    # for s in seeds:
    #     exec_config.seed = s
    run_tomo_cedar(exec_config)

