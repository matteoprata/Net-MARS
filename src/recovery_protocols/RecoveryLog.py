import numpy as np
import pandas as pd
from enum import Enum


class DFColumns(Enum):
    ITER = "iter"
    REPAIRS = "repairs"
    EVENTS = "event"
    TOTAL_FLOW = "flow_cum"
    MAXIMUM_FLOW = "total_flow"


class RecoveryLog:
    def __init__(self, algorithm, config, UB=500):
        self.UB = UB
        self.df = pd.DataFrame(np.zeros((UB, 5)), columns=[DFColumns.ITER.value,
                                                           DFColumns.REPAIRS.value,
                                                           DFColumns.EVENTS.value,
                                                           DFColumns.TOTAL_FLOW.value,
                                                           DFColumns.MAXIMUM_FLOW.value])

        self.df[DFColumns.ITER.value] = self.df[DFColumns.ITER.value].astype(int)
        self.df[DFColumns.REPAIRS.value] = self.df[DFColumns.REPAIRS.value].astype(str)

        self.flow_per_demand = {}
        self.n_repairs = 0
        self.n_monitors = 0
        self.n_packet_monitoring = 0
        self.time = 0

        self.monitors = []

    def get_iter(self, t):
        return self.df.loc[t, DFColumns.ITER.value]

    def get_repair(self, t):
        return self.df.loc[t, DFColumns.REPAIRS.value]

    def get_event(self, t):
        return self.df.loc[t, DFColumns.EVENTS.value]

    def get_total_flow(self, t):
        return self.df.loc[t, DFColumns.TOTAL_FLOW.value]

    def get_maximum_flow(self, t):
        return self.df.loc[t, DFColumns.MAXIMUM_FLOW.value]

    def show_log(self, up_to=None):
        up_to = self.df.shape[0] if up_to is None else up_to
        return self.df[:up_to]

    def add_demand_flow(self, demand_edge, quantity):
        if demand_edge not in self.flow_per_demand:
            self.flow_per_demand[demand_edge] = np.zeros(shape=self.UB)
        self.flow_per_demand[demand_edge][self.time] = quantity

    def __add_iter(self):
        self.df.loc[self.time, DFColumns.ITER.value] = self.time

    def add_repairs(self, elements):
        for el in elements:
            self.df.loc[self.time, DFColumns.REPAIRS.value] = str(el)
            # self.step()

    def add_maximum_flow(self, maximum_flow):
        self.df.loc[self.time, DFColumns.MAXIMUM_FLOW.value] = maximum_flow

    def add_total_flow(self, total_flow):
        self.df.loc[self.time, DFColumns.TOTAL_FLOW.value] = total_flow

    def add_event(self, event):
        self.df.loc[self.time, DFColumns.EVENTS.value] = event

    def add_n_packet_monitoring(self, n_packet_monitoring):
        self.n_packet_monitoring += n_packet_monitoring

    def add_monitors(self, monitors):
        for mon in monitors:
            if mon not in self.monitors:
                self.monitors.append(mon)

    def step(self):
        self.__add_iter()
        self.time += 1

    def empty_vec(self):
        return np.zeros(shape=self.UB)