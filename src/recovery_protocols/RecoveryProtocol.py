
class RecoveryProtocol:
    """ The prototype to inherit to create a recovery protocol. """

    file_name = None
    """ Name of the file containing the stats. Could be equal to plot name. """

    plot_name = None
    """ Name of the algorithm on the plot. """

    mode_path_repairing = None
    mode_path_choosing_repair = None
    mode_monitoring = None
    mode_monitoring_type = None

    plot_marker = None
    plot_color_curve = None

    def __init__(self, config):
        self.config = config

    def run(self):
        pass
