import os.path

import numpy as np
import pandas as pd
import code
import matplotlib.pyplot as plt
import rsome
from rsome import ro
from rsome import grb_solver as grb

import graphs
import utils
from pds import PDS


class WaterNet:
    """
    This class is structured for linear optimization of the wds operation
    While the class in wds.py is more general
    """

    def __init__(self, data_folder):
        self.data_folder = data_folder

        self.combs = pd.read_csv(os.path.join(self.data_folder, 'combs.csv'), index_col=0)
        self.tanks = pd.read_csv(os.path.join(self.data_folder, 'tanks.csv'), index_col=0)
        self.demands = pd.read_csv(os.path.join(self.data_folder, 'demands.csv'), index_col=0)
        self.tariffs = pd.read_csv(os.path.join(self.data_folder, 'tariffs.csv'), index_col=0)

        self.n_combs = len(self.combs)
        self.n_tanks = len(self.tanks)

        self.tanks['init_vol'] = self.level_to_vol(self.tanks['diameter'], self.tanks['init_level'])
        self.tanks['min_vol'] = self.level_to_vol(self.tanks['diameter'], self.tanks['min_level'])
        self.tanks['max_vol'] = self.level_to_vol(self.tanks['diameter'], self.tanks['max_level'])

    def get_tank_demand(self, tank_id):
        return self.demands.loc[:, tank_id]

    @staticmethod
    def level_to_vol(diameter, level):
        return level * np.pi * (diameter ** 2) / 4


class Model:
    def __init__(self, pds_data: str, wds_data: str, t: int):
        self.pds_data = pds_data
        self.wds_data = wds_data
        self.t = t

        self.pds = PDS(self.pds_data)
        self.wds = WaterNet(self.wds_data)

        self.model = ro.Model()
        self.x = self.declare_vars()

    def declare_vars(self):
        gen_p = self.model.dvar((self.pds.n_bus, self.t))  # active power generation
        gen_q = self.model.dvar((self.pds.n_bus, self.t))  # reactive power generation
        psh_y = self.model.dvar((self.pds.n_psh, self.t))  # psh_y = pumped storage hydropower injection
        psh_h = self.model.dvar((self.pds.n_psh, self.t))  # psh_y = pumped storage hydropower consumption

        v = self.model.dvar((self.pds.n_bus, self.t))  # buses squared voltage
        I = self.model.dvar((self.pds.n_lines, self.t))  # lines squared current flow
        p = self.model.dvar((self.pds.n_lines, self.t))  # lines active power flow
        q = self.model.dvar((self.pds.n_lines, self.t))  # lines reactive power flow

        self.model.st(gen_p >= 0)
        self.model.st(gen_q >= 0)
        self.model.st(psh_y >= 0)
        self.model.st(psh_h >= 0)
        self.model.st(I >= 0)

        x_pumps = self.model.dvar((self.wds.n_combs, self.t), vtype='B')
        vol = self.model.dvar((self.wds.n_tanks, self.t))  # tanks volume

        self.model.st(0 <= x_pumps)
        self.model.st(x_pumps <= 1)
        self.model.st(vol >= 0)

        return {'gen_p': gen_p, 'gen_q': gen_q, 'psh_y': psh_y, 'psh_h': psh_h, 'v': v, 'I': I, 'p': p, 'q': q,
                'pumps': x_pumps, 'vol': vol}

    def get_wds_cost(self):
        wds_power = self.wds.combs.loc[:, "total_power"].values.reshape(1, -1) @ self.x['pumps']
        wds_cost = (self.wds.tariffs.values.T * wds_power).sum()
        return wds_cost

    def get_pds_cost(self):
        pds_cost = (self.pds.gen_mat @ (self.pds.pu_to_kw * self.x['gen_p']) * self.pds.tariffs.values).sum().sum()
        return pds_cost

    def build_water_problem(self):
        wds_cost = self.get_wds_cost()
        self.objective_func(wds_cost, 0)
        self.one_comb_only()
        self.mass_balance()

    def build_combined_problem(self, x_pumps=None):
        wds_cost = 0
        pds_cost = self.get_pds_cost()
        self.objective_func(wds_cost, pds_cost)
        self.bus_balance(x_pumps=x_pumps)
        self.energy_conservation()
        self.voltage_bounds()
        self.power_flow_constraint()

        if x_pumps is not None:
            self.model.st(self.x['pumps'] - x_pumps == 0)

        self.one_comb_only()
        self.mass_balance()

    def objective_func(self, wds_obj, pds_obj):
        self.model.min(wds_obj + pds_obj)

    def bus_balance(self, x_pumps):
        r = utils.get_connectivity_mat(self.pds.lines, from_col='from_bus', to_col='to_bus', direction='in',
                                       param='r_pu')
        x = utils.get_connectivity_mat(self.pds.lines, from_col='from_bus', to_col='to_bus', direction='in',
                                       param='x_pu')
        a = utils.get_connectivity_mat(self.pds.lines, from_col='from_bus', to_col='to_bus')

        bus_pumps = self.pds.construct_bus_pumps_mat()
        power_mat = self.wds.combs[[_ for _ in self.wds.combs.columns if _.startswith('power')]].fillna(0).values.T
        power_mat = (power_mat * 1000) / (self.pds.power_base_mva * 10 ** 6)
        if x_pumps is not None:
            pumps_power = bus_pumps @ power_mat @ x_pumps
        else:
            pumps_power = bus_pumps @ power_mat @ self.x['pumps']

        self.model.st(self.pds.gen_mat @ self.x['gen_p'] + a @ self.x['p']
                      - r @ self.x['I']
                      - self.pds.dem_active.values - pumps_power
                      + self.pds.bus.loc[:, 'G'].values @ self.x['v']
                      == 0)

        self.model.st(self.pds.gen_mat @ self.x['gen_q'] + a @ self.x['q']
                      - x @ self.x['I']
                      - self.pds.dem_reactive.values
                      + self.pds.bus.loc[:, 'B'].values @ self.x['v']
                      == 0)

    def voltage_bounds(self):
        self.model.st(self.x['v'] - self.pds.bus['Vmax_pu'].values.reshape(-1, 1) <= 0)
        self.model.st(self.pds.bus['Vmin_pu'].values.reshape(-1, 1) - self.x['v'] <= 0)

    def power_flow_constraint(self):
        for t in range(self.t):
            for line in range(self.pds.n_lines):
                b_id = self.pds.lines.loc[line, 'to_bus']
                self.model.st(rsome.rsocone(self.x['p'][line, t] + self.x['q'][line, t],
                                            self.x['v'][b_id, t],
                                            self.x['I'][line, t]))

    def energy_conservation(self):
        r = self.pds.lines['r_pu'].values.reshape(1, -1)
        x = self.pds.lines['x_pu'].values.reshape(1, -1)
        a = utils.get_connectivity_mat(self.pds.lines, from_col='from_bus', to_col='to_bus')
        self.model.st(a.T @ self.x['v']
                      + 2 * ((self.x['p'].T * r).T + (self.x['q'].T * x).T)
                      - (self.x['I'].T * (r ** 2 + x ** 2)).T
                      == 0)

    def one_comb_only(self):
        for station in self.wds.combs['station'].unique():
            idx = self.wds.combs.loc[self.wds.combs['station'] == station].index.to_list()
            self.model.st(sum([self.x['pumps'][_, :] for _ in idx]) <= 1)

    def construct_tank_mat(self):
        mat = np.diag(-np.ones(self.t))
        rows, cols = np.indices((self.t, self.t))
        row_vals = np.diag(rows, k=-1)
        col_vals = np.diag(cols, k=-1)
        mat[row_vals, col_vals] = 1
        return mat

    def construct_combs_mat(self, tank_id, param='flow'):
        mat = np.zeros((self.t, self.wds.n_combs * self.t))
        inflow_idx = self.wds.combs.loc[self.wds.combs['to'] == tank_id].index.to_list()
        outflow_idx = self.wds.combs.loc[self.wds.combs['from'] == tank_id].index.to_list()
        for i in range(self.t):
            inflow_cols = [col + i * self.wds.n_combs for col in inflow_idx]
            outflow_cols = [col + i * self.wds.n_combs for col in outflow_idx]
            mat[i, inflow_cols] = 1
            mat[i, outflow_cols] = -1

        param_array = np.tile(self.wds.combs[param].to_numpy(), self.t)
        mat = mat * param_array
        return mat

    def mass_balance(self):
        lhs = np.zeros((self.wds.n_tanks * self.t, (self.x['pumps'].shape[0] + self.x['vol'].shape[0]) * self.t))
        rhs = np.zeros((self.wds.n_tanks * self.t, 1))

        for i, (tank_id, row) in enumerate(self.wds.tanks.iterrows()):
            flows_mat = self.construct_combs_mat(tank_id, param='flow')
            lhs[self.t * i: self.t * (i + 1), : - self.wds.n_tanks * self.t] = flows_mat

            tank_mat = self.construct_tank_mat()
            lhs[self.t * i: self.t * (i + 1),
            (self.wds.n_combs + i) * self.t: (self.wds.n_combs + i + 1) * self.t] = tank_mat

            dem = self.wds.get_tank_demand(tank_id).to_numpy()[:, np.newaxis]
            rhs[self.t * i: self.t * (i + 1)] = dem
            rhs[self.t * i] = rhs[self.t * i] - self.wds.tanks.loc[tank_id, 'init_vol']

            self.model.st(self.x['vol'][i, :] <= self.wds.tanks.loc[tank_id, 'max_vol'])
            self.model.st(self.x['vol'][i, :] >= self.wds.tanks.loc[tank_id, 'min_vol'])
            self.model.st(self.x['vol'][i, -1] - self.wds.tanks.loc[tank_id, 'init_vol'] >= 0)

        self.model.st(lhs[:, : - self.wds.n_tanks * self.t] @ self.x['pumps'].T.flatten()
                      + lhs[:, self.wds.n_combs * self.t:] @ self.x['vol'].flatten()
                      == np.squeeze(rhs))

    def solve(self):
        self.model.solve(grb, display=False)
        obj, status = self.model.solution.objval, self.model.solution.status
        wds_cost, pds_cost = self.get_systemwise_costs()
        print(obj, status, wds_cost, pds_cost)

    def get_systemwise_costs(self):
        wds_power = self.wds.combs.loc[:, "total_power"].values.reshape(1, -1) @ self.x['pumps'].get()
        wds_cost = (self.wds.tariffs.values.T * wds_power).sum()
        pds_cost = np.sum(self.pds.gen_mat @ (self.pds.pu_to_kw * self.x['gen_p'].get()) * self.pds.tariffs.values)
        return wds_cost, pds_cost


def solve_water():
    model = Model(pds_data=os.path.join('data', 'pds'), wds_data=os.path.join('data', 'wds_wells'), t=24)
    model.build_water_problem()
    model.solve()
    return model


def solve_combined(x_pumps=None):
    model = Model(pds_data=os.path.join('data', 'pds'), wds_data=os.path.join('data', 'wds_wells'), t=24)
    model.build_combined_problem(x_pumps)
    model.solve()
    return model