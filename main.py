import os

import rsome
from rsome import ro
from rsome import grb_solver as grb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import opt
import graphs
from pds import PDS
from wds import WDS

plt.close("all")


class Opt:
    def __init__(self, pds_data: str, wds_data: str, T: int):
        self.pds_data = pds_data
        self.wds_data = wds_data
        self.T = T

        self.pds, self.wds = self.init_distribution_systems()
        self.model = ro.Model()
        self.x = self.declare_vars()
        self.build()

    def init_distribution_systems(self):
        pds = PDS(self.pds_data)
        wds = WDS(self.wds_data)
        return pds, wds

    def declare_vars(self):
        gen_p = self.model.dvar((self.pds.n_bus, self.T))
        gen_q = self.model.dvar((self.pds.n_bus, self.T))
        psh_y = self.model.dvar((self.pds.n_psh, self.T))  # psh_y = pumped storage hydropower injection
        psh_h = self.model.dvar((self.pds.n_psh, self.T))  # psh_y = pumped storage hydropower consumption

        v = self.model.dvar((self.pds.n_bus, self.T))  # buses voltage
        I = self.model.dvar((self.pds.n_lines, self.T))  # lines squared current flow
        p = self.model.dvar((self.pds.n_lines, self.T))  # active power flow from\to bus
        q = self.model.dvar((self.pds.n_lines, self.T))  # reactive power flow from\to bus

        self.model.st(gen_p >= 0)
        self.model.st(gen_q >= 0)
        self.model.st(psh_y >= 0)
        self.model.st(psh_h >= 0)
        self.model.st(I >= 0)

        pump_p = self.model.dvar((self.wds.n_pumps, self.T))
        return {'gen_p': gen_p, 'gen_q': gen_q, 'psh_y': psh_y, 'psh_h': psh_h, 'v': v, 'I': I, 'p': p, 'q': q}

    def build(self):
        self.objective_func()
        self.bus_balance()
        self.energy_conservation()
        self.voltage_bounds()
        self.power_flow_constraint()

    def objective_func(self):
        self.model.min((self.pds.gen_mat @ self.x['gen_p'] @ self.pds.grid_tariff.values).sum().sum()
                       + (self.pds.psh['fill_tariff'].values @ self.x['psh_y']).sum()
                       )

    def bus_balance(self):
        r = self.pds.bus_lines_mat(direction='in', param='r_kohm')
        x = self.pds.bus_lines_mat(direction='in', param='x_kohm')
        a = self.pds.bus_lines_mat()

        self.model.st(self.pds.gen_mat @ self.x['gen_p'] + a @ self.x['p']
                      - r @ self.x['I']
                      - self.pds.dem_active.values
                      + self.pds.bus.loc[:, 'G'].values @ self.x['v']
                      == 0)

        self.model.st(self.pds.gen_mat @ self.x['gen_q'] + a @ self.x['q']
                      - x @ self.x['I']
                      - self.pds.dem_reactive.values
                      + self.pds.bus.loc[:, 'B'].values @ self.x['v']
                      == 0)

    def energy_conservation(self):
        r = self.pds.lines['r_ohm'].values.reshape(1, -1)
        x = self.pds.lines['x_ohm'].values.reshape(1, -1)
        a = self.pds.bus_lines_mat()

        self.model.st(a.T @ self.x['v'] * self.pds.nominal_voltage_kv
                      + 2 * ((self.x['p'].T * r).T + (self.x['q'].T * x).T)
                      - (self.x['I'].T * (r ** 2 + x ** 2)).T
                      == 0)

    def voltage_bounds(self):
        nom_v = self.pds.nominal_voltage_kv
        self.model.st(self.x['v'] - self.pds.bus['Vmax_pu'].values.reshape(-1, 1) * nom_v <= 0)
        self.model.st(self.pds.bus['Vmin_pu'].values.reshape(-1, 1) * nom_v - self.x['v'] <= 0)

    def power_flow_constraint(self):
        for t in range(self.T):
            for l in range(self.pds.n_lines):
                b_id = self.pds.lines.loc[l, 'to_bus']
                self.model.st(rsome.rsocone(self.x['p'][l, t] + self.x['q'][l, t],
                                            self.x['v'][b_id, t],
                                            self.x['I'][l, t]))

    def solve(self):
        self.model.solve(display=False)
        obj, status = self.model.solution.objval, self.model.solution.status
        print(obj, status)

    def extract_results(self, x, elem_type, series_type='time', elem_idx=None, t_idx=None, dec=1, factor=1):
        n = {'nodes': self.pds.n_bus, 'lines': self.pds.n_lines}
        if series_type == 'time':
            values = {t: self.x[x].get()[elem_idx, t] * factor for t in range(self.T)}

        if series_type == 'elements':
            values = {i: self.x[x].get()[i, t_idx] * factor for i in range(n[elem_type])}

        return {t: round(val, dec) for t, val in values.items()}

    def plot_results(self, t, net_coords):
        f = 1 / self.pds.nominal_voltage_kv
        nodes_vals = self.extract_results(x='v', elem_type='nodes', series_type='elements', t_idx=0, dec=2, factor=f)
        edges_vals = self.extract_results(x='p', elem_type='lines', series_type='elements', t_idx=0)
        gr = graphs.OptGraphs(self.pds)
        gr.pds_graph(edges_values=edges_vals, nodes_values=nodes_vals, net_coords=net_coords)

        graphs.time_series(x=range(self.T), y=self.x['gen_p'].get()[t, :])


if __name__ == "__main__":
    PDS_DATA = os.path.join('data', 'pds')
    WDS_DATA = os.path.join('data', 'wds')
    opt = Opt(pds_data=PDS_DATA, wds_data=WDS_DATA, T=24)
    opt.solve()
    opt.plot_results(t=0, net_coords=graphs.IEEE33_POS)

    plt.show()
