
import src.constants as co


class RecoveryProtocol:
    """ The prototype to inherit to create a recovery protocol. """

    file_name: str = None
    """ Name of the file containing the stats. Could be equal to plot name. """

    plot_name: str = None
    """ Name of the algorithm on the plot. """

    mode_path_repairing: co.ProtocolRepairingPath = None
    """ Modality to chose the path to repair.  """

    mode_path_choosing_repair: co.ProtocolPickingPath = None
    """ Modality to pick the path to repair among those available.  """

    mode_monitoring: co.ProtocolMonitorPlacement = None
    """ Modality to place monitors.  """

    mode_monitoring_type: co.PriorKnowledge = None
    """ Type of knowledge to acquire. """

    plot_marker = None
    """ Marker of the algorithm on the plot. """

    plot_color_curve = None
    """ Color of the algorithm on the plot. """

    def __init__(self, config):
        self.config = config

    def run(self):
        pass
