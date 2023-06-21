import os
from rsome import ro
from rsome import grb_solver as grb
import pandas as pd
import numpy as np

from pds import PDS
from wds import WDS

PDS_DATA = os.path.join('data', 'pds')
WDS_DATA = os.path.join('data', 'wds')


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
        gen_p = self.model.dvar((len(self.pds.generators), self.T))
        gen_q = self.model.dvar((len(self.pds.generators), self.T))
        psh_y = self.model.dvar((self.pds.n_psh, self.T))  # psh_y = pumped storage hydropower injection
        psh_h = self.model.dvar((self.pds.n_psh, self.T))  # psh_y = pumped storage hydropower consumption

        v = self.model.dvar((self.pds.n_bus, self.T))  # buses voltage
        I = self.model.dvar((self.pds.n_bus, self.T))  # lines squared current flow
        p = self.model.dvar((self.pds.n_bus, self.T))  # active power flow from\to bus
        q = self.model.dvar((self.pds.n_bus, self.T))  # reactive power flow from\to bus

        self.model.st(gen_p >= 0)
        self.model.st(gen_q >= 0)
        self.model.st(psh_y >= 0)
        self.model.st(psh_h >= 0)
        self.model.st(v >= 0)
        self.model.st(I >= 0)

        pump_p = self.model.dvar((self.wds.n_pumps, self.T))
        return {'gen_p': gen_p, 'gen_q': gen_q, 'psh_y': psh_y, 'psh_h': psh_h, 'v': v, 'I': I, 'p': p, 'q': q}

    def build(self):
        self.objective_func()
        self.generators_balance()
        self.bus_balance()

    def objective_func(self):
        self.model.min((self.pds.grid_tariff.values @ self.x['gen_p']).sum()
                       + (self.pds.psh['fill_tariff'].values @ self.x['psh_y']).sum())

    def generators_balance(self):
        # generators balance - eq 4-5
        gen_idx = self.pds.generators.index.to_list()
        self.model.st(self.x['gen_p'] - self.x['p'][gen_idx, :]
                      - self.pds.bus.loc[gen_idx, 'G'].values @ self.x['v'][gen_idx, :] == 0)

        self.model.st(self.x['gen_q'] - self.x['q'][gen_idx, :]
                      - self.pds.bus.loc[gen_idx, 'B'].values @ self.x['v'][gen_idx, :] == 0)

    def bus_balance(self):
        r = self.pds.get_connectivity_mat(param='r_ohm')
        x = self.pds.get_connectivity_mat(param='x_ohm')

        self.model.st(self.pds.A.T @ self.x['p'] - r @ self.x['I'] - self.pds.A @ self.x['p']
                      - self.pds.dem_active.values + self.pds.bus.loc[:, 'G'].values @ self.x['v'] == 0)

        self.model.st(self.pds.A.T @ self.x['q'] - x @ self.x['I'] - self.pds.A @ self.x['q']
                      - self.pds.dem_reactive_power.values + self.pds.bus.loc[:, 'B'].values @ self.x['v'] == 0)

    def energy_conservation(self):
        r = self.pds.get_connectivity_mat(param='r_ohm')
        x = self.pds.get_connectivity_mat(param='x_ohm')
        self.model.st(-2 * (r @ self.x['p']) + ((r ** 2 + x ** 2) @ self.x['I']))

    def solve(self):
        self.model.solve(display=False)
        obj, status = self.model.solution.objval, self.model.solution.status
        print(obj, status)
        df = (pd.DataFrame({'from': (self.pds.A @ self.x['p'].get())[:, 0],
              'to': (self.pds.A.T @ self.x['p'].get())[:, 0]}))
        df['d'] = df['to'] - df['from']
        print(df)
        print(df.sum())
        print(self.x['gen_p'].get())


if __name__ == "__main__":
    opt = Opt(pds_data=PDS_DATA, wds_data=WDS_DATA, T=24)
    # opt.solve()
    print(opt.pds.A - opt.pds.A.T)