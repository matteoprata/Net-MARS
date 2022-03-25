
import numpy as np
import itertools
import argparse
from multiprocessing import Pool
import os

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


def run_var_seed_dis(seed, dis, budget, nnodes, flowpp, rep_mode, pick_mode, monitor_placement, is_parallel=False):
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
    config.monitors_budget_residual = budget
    config.n_demand_clique = nnodes
    config.demand_capacity = flowpp
    config.repairing_mode = rep_mode
    config.picking_mode = pick_mode

    config.protocol_monitor_placement = monitor_placement
    if config.protocol_monitor_placement == co.ProtocolMonitorPlacement.ORACLE:
        config.is_oracle_baseline = True

    config_details = print_configuration(config)

    fname = "exp-s|{}-g|{}-np|{}-dc|{}-spc|{}-alg|{}-bud|{}-pbro|{}-rep|{}-pik|{}-mop|{}.csv".format(config.seed,
                                                                                                     config.graph_dataset.name,
                                                                                                     config.n_demand_clique,
                                                                                                     int(config.demand_capacity),
                                                                                                     config.supply_capacity,
                                                                                                     config.algo_name,
                                                                                                     config.monitors_budget,
                                                                                                     config.destruction_quantity,
                                                                                                     config.repairing_mode.value,
                                                                                                     config.picking_mode.value,
                                                                                                     config.protocol_monitor_placement.value)

    if not os.path.exists("data/experiments/"+fname):
        print()
        print("NOW running...\n\n", config_details, "\n")

        set_seeds(config.seed)

        if config.mute_log:
            block_print()

        stats = main_cedar_setup.run(config)
        enable_print()

        if stats is not None:
            save_stats_as_df_ph1(stats, fname)
    else:
        print()
        print("THIS already existed...\n", fname, "\n")


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

    # seeds = [999, 798, 678, 543, 11, 3, 5]
    # seeds = [66, 1000, 33, 34, 56, 979, 349]  # regina elena
    # seeds = [77, 78, 79, 90, 400, 50, 55, 999, 798, 678, 543, 979, 1000, 5221]  # range(30, 39)
    seeds = [90, 400, 50, 999, 798, 678, 543, 979]

    dis_uni = [.05, .15, .3, .5, .7]
    budget_n_monitor = [10]
    npairs = [8]
    flowpp = [10.0]

    reps = [
            # co.ProtocolRepairingPath.SHORTEST,
            co.ProtocolRepairingPath.MAX_BOT_CAP,
            # co.ProtocolRepairingPath.MIN_COST_BOT_CAP,
            ]  # co.ProtocolRepairingPath.IP

    pick = [# co.ProtocolPickingPath.RANDOM,
            co.ProtocolPickingPath.MIN_COST_BOT_CAP,
            # co.ProtocolPickingPath.MAX_BOT_CAP
            ]

    mplacement = [
                   co.ProtocolMonitorPlacement.STEP_BY_STEP,
                   # co.ProtocolMonitorPlacement.BUDGET_W_REPLACEMENT,
                   # co.ProtocolMonitorPlacement.ORACLE
    ]

    processes = []
    for execution in itertools.product(seeds, dis_uni, budget_n_monitor, npairs, flowpp, reps, pick, mplacement, [True]):
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
    seeds = [90, 400, 50, 999, 798, 678, 543, 979]
    dis_uni = [.05, .15, .3, .5, .7]

    # i rep, j pik, k mop
    algos =  [("CEDARNEW", [i, j, k]) for i in range(2, 3) for j in range(2, 3) for k in [3]]
    algos +=  [("CEDARNEW", [i, j, k]) for i in range(0, 4) for j in range(2, 3) for k in [1]]

    # #
    # algos += [("CEDARNEW", [i, j, k]) for i in range(2, 3) for j in range(2, 3) for k in [3]]

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
    plotting_data()
    # run_var_seed_dis(seed=543, dis=.5, budget=10, nnodes=8, flowpp=10,
    #                  rep_mode=co.ProtocolRepairingPath.MIN_COST_BOT_CAP,
    #                  pick_mode=co.ProtocolPickingPath.MIN_COST_BOT_CAP,
    #                  monitor_placement=co.ProtocolMonitorPlacement.BUDGET_W_REPLACEMENT,
    #                  )
