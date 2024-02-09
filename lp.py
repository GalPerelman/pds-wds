import os.path

import gurobipy.gurobipy
import numpy as np
import pandas as pd
import code
import warnings
import rsome
from rsome import ro
from rsome import grb_solver as grb

import graphs
import utils
from pds import PDS


class WDS:
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
        self.n_stations = self.combs['station'].nunique()
        self.n_pumps = len([_ for _ in self.combs.columns if _.startswith('pump_')])
        self.pumps_combs = self.get_pumps_combs_mat()

        self.tanks['init_vol'] = self.level_to_vol(self.tanks['diameter'], self.tanks['init_level'])
        self.tanks['min_vol'] = self.level_to_vol(self.tanks['diameter'], self.tanks['min_level'])
        self.tanks['max_vol'] = self.level_to_vol(self.tanks['diameter'], self.tanks['max_level'])

    def get_tank_demand(self, tank_id):
        return self.demands.loc[:, tank_id]

    @staticmethod
    def level_to_vol(diameter, level):
        return level * np.pi * (diameter ** 2) / 4

    def get_pumped_vol(self, x_pumps):
        vol = self.combs.loc[:, "flow"].values.reshape(1, -1) @ x_pumps
        return vol

    def get_pumps_combs_mat(self):
        """
        Build a connectivity matrix between individual pumps to pump combinations
        """
        mat = self.combs[[_ for _ in self.combs.columns if _.startswith('pump_')]].fillna(0).values.T
        return mat

    def factorize_demands(self, factor=1):
        self.demands *= factor


class Optimizer:
    def __init__(self, pds_data: str, wds_data: str, scenario=None, display=True):
        self.pds_data = pds_data
        self.wds_data = wds_data
        self.scenario = scenario
        self.display = display

        self.t = 24
        self.outage_lines = []

        self.pds = PDS(self.pds_data)
        self.wds = WDS(self.wds_data)

        if self.scenario is not None:
            self.assign_scenario()

        self.model = ro.Model()
        self.x = self.declare_vars()

        self.objective = None
        self.status = None

    def assign_scenario(self):
        self.pds.factorize_demands(active_factor=self.scenario.pds_demand_factor,
                                   reactive_factor=self.scenario.pds_demand_factor)
        self.wds.factorize_demands(self.scenario.wds_demand_factor)
        self.pds.bus.loc[self.pds.bus['type'] == 'pv', 'max_gen_kw'] *= self.scenario.pv_factor
        self.wds.tanks['init_level'] *= self.scenario.tanks_state.T
        self.pds.bus.loc[self.pds.bus['max_storage'] > 0, 'init_storage'] *= self.scenario.batteries_state.T
        self.t = self.scenario.t
        self.outage_lines = self.scenario.outage_lines

        # adjust all pds and wds data to be within the scenario duration
        self.pds.dem_active = self.pds.dem_active.iloc[:, :self.t]
        self.pds.dem_reactive = self.pds.dem_reactive.iloc[:, :self.t]
        self.pds.tariffs = self.pds.tariffs.iloc[:, :self.t]
        self.wds.demands = self.wds.demands.iloc[:self.t]
        self.wds.tariffs = self.wds.tariffs.iloc[:self.t]

    def declare_vars(self):
        gen_p = self.model.dvar((self.pds.n_bus, self.t))  # active power generation
        gen_q = self.model.dvar((self.pds.n_bus, self.t))  # reactive power generation
        bat_p = self.model.dvar((self.pds.n_bus, self.t))  # batteries charge - positive when filling
        bat_e = self.model.dvar((self.pds.n_bus, self.t))  # batteries state
        penalty_p = self.model.dvar((self.pds.n_bus, self.t))  # penalty for gap between demand and supply
        penalty_q = self.model.dvar((self.pds.n_bus, self.t))  # penalty for gap between demand and supply

        v = self.model.dvar((self.pds.n_bus, self.t))  # buses squared voltage
        I = self.model.dvar((self.pds.n_lines, self.t))  # lines squared current flow
        p = self.model.dvar((self.pds.n_lines, self.t))  # lines active power flow
        q = self.model.dvar((self.pds.n_lines, self.t))  # lines reactive power flow

        self.model.st(gen_p >= 0)
        self.model.st(gen_q >= 0)
        self.model.st(I >= 0)
        self.model.st(penalty_p >= 0)
        self.model.st(penalty_q >= 0)

        x_pumps = self.model.dvar((self.wds.n_combs, self.t))
        vol = self.model.dvar((self.wds.n_tanks, self.t))  # tanks volume

        self.model.st(0 <= x_pumps)
        self.model.st(x_pumps <= 1)
        self.model.st(vol >= 0)

        return {'gen_p': gen_p, 'gen_q': gen_q, 'bat_p': bat_p, 'bat_e': bat_e, 'v': v, 'I': I, 'p': p, 'q': q,
                'penalty_p': penalty_p, 'penalty_q': penalty_q,
                'pumps': x_pumps, 'vol': vol
                }

    def get_wds_cost(self):
        wds_power = self.wds.combs.loc[:, "total_power"].values.reshape(1, -1) @ self.x['pumps']
        wds_cost = (self.wds.tariffs.values.T * wds_power).sum()
        return wds_cost

    def get_pds_cost(self):
        # constant cost term - like purchasing from grid
        const_term = (self.pds.gen_mat @ (self.pds.pu_to_kw * self.x['gen_p']) * self.pds.tariffs.values).sum().sum()

        # generation cost term - fuel cost for generators, usually 0 for grid and renewable
        # NOTE THAT C IS MULTIPLIED BY T - NUMBER OF TIME STEPS
        # ASSUMING GENERATORS ARE NOT TURNED OFF - THUS FIXED COST IS FOR EVERY TIME STEP
        generated_kw = self.x['gen_p'] * self.pds.pu_to_kw
        generation_cost = (rsome.sumsqr(((self.pds.bus['a'].values.reshape(-1, 1) ** 0.5) * generated_kw).flatten())
                           + (self.pds.bus['b'].values.reshape(-1, 1) * generated_kw).sum()
                           + (self.pds.bus['c'].values.reshape(-1, 1) * self.t).sum()
                           )

        return const_term + generation_cost

    def build_water_problem(self, obj, w=None):
        if obj == "cost":
            wds_cost = self.get_wds_cost()
            self.cost_objective_func(wds_cost, 0)
        elif obj == "emergency":
            if w is None:
                w = np.ones((self.wds.n_pumps, self.t))
            # version 1
            # (1 x n_combs) @ (n_combs x T) = (1 x T)
            # wds_power = self.wds.combs.loc[:, "total_power"].values.reshape(1, -1) @ self.x['pumps']
            # power = (tw * wds_power).sum()
            # self.model.min(power)

            # version 2
            # (n_bus x n_pumps) @ (n_pumps x n_combs) @ (n_combs x T)
            # power_mat = self.wds.combs[[_ for _ in self.wds.combs.columns if _.startswith('power')]].fillna(0).values.T
            # power_mat = (power_mat * 1000) / (self.pds.power_base_mva * 10 ** 6)
            # pumps_power = self.pds.pumps_bus.values @ power_mat @ self.x['pumps']
            # w - weights of penalties, larger values means larger reduction in pumping
            # w should be with size of (n_bus x T)
            # pumps_penalized_power = w @ pumps_power

            # version 3
            # (T x n_pumps) @ (n_pumps x n_combs) @ (n_combs x T)
            wds_power = self.wds.pumps_combs @ self.x['pumps']
            pumps_penalized_power = w.T @ wds_power

            self.model.min(pumps_penalized_power.sum())

        self.one_comb_only()
        self.mass_balance()

    def build_combined_problem(self, x_pumps=None):
        wds_cost = 0
        pds_cost = self.get_pds_cost()
        self.cost_objective_func(wds_cost, pds_cost)
        self.power_generation_bounds()
        self.batteries_bounds()
        self.batteries_balance()
        self.penalty_bounds(ub=0)
        self.bus_balance(x_pumps=x_pumps)
        self.energy_conservation()
        self.voltage_bounds()
        self.power_flow_constraint()

        for line_idx in self.outage_lines:
            self.disable_power_line(line_idx)

        if x_pumps is not None:
            self.model.st(self.x['pumps'] - x_pumps == 0)

        self.one_comb_only()
        self.mass_balance()

    def build_combined_resilience_problem(self, x_pumps=None):
        self.emergency_objective()
        self.power_generation_bounds()
        self.batteries_bounds()
        self.batteries_balance()
        self.bus_balance(x_pumps=x_pumps)
        self.energy_conservation()
        self.voltage_bounds()
        self.power_flow_constraint()

        for line_idx in self.outage_lines:
            self.disable_power_line(line_idx)

        if x_pumps is not None:
            self.model.st(self.x['pumps'] - x_pumps == 0)

        self.one_comb_only()
        self.mass_balance()

    def emergency_objective(self):
        """
        Objective function for resilience optimization
        Minimization of the gap between demand and supply
        The bus power balance is formulated as soft constraint
        Objective units are kWhr - penalty is power (kw) which is summed over time
        """
        obj = (self.x['penalty_p'] * self.pds.bus_criticality.iloc[:, :self.t].values * self.pds.pu_to_kw).sum()
        self.model.min(obj)

    def cost_objective_func(self, wds_obj, pds_obj):
        self.model.min(wds_obj + pds_obj)

    def power_generation_bounds(self):
        min_power = np.multiply(self.pds.bus['min_gen_kw'].values, self.pds.max_gen_profile.iloc[:, :self.t].T).T.values
        max_power = np.multiply(self.pds.bus['max_gen_kw'].values, self.pds.max_gen_profile.iloc[:, :self.t].T).T.values
        self.model.st(- self.x['gen_p'] + min_power <= 0)
        self.model.st(self.x['gen_p'] - max_power <= 0)

    def batteries_bounds(self):
        # charging rate constraint
        self.model.st(self.x['bat_p'] - self.pds.bus['max_power'].values.reshape(-1, 1) <= 0)
        self.model.st(- self.pds.bus['max_power'].values.reshape(-1, 1) - self.x['bat_p'] <= 0)

        # energy storage constraint
        self.model.st(self.x['bat_e'] - self.pds.bus['max_storage'].values.reshape(-1, 1) <= 0)
        self.model.st(- self.x['bat_e'] + self.pds.bus['min_storage'].values.reshape(-1, 1) <= 0)

    def batteries_balance(self):
        mat = np.triu(np.ones((self.t, self.t)))
        init_mat = np.zeros((self.pds.n_bus, self.t))
        init_mat[:, 0] = self.pds.bus['init_storage'].values
        self.model.st((self.x['bat_p'] @ mat) + init_mat - self.x['bat_e'] == 0)

    def bus_balance(self, x_pumps):
        r = utils.connectivity_mat(self.pds.lines, from_col='from_bus', to_col='to_bus', direction='in', param='r_pu')
        x = utils.connectivity_mat(self.pds.lines, from_col='from_bus', to_col='to_bus', direction='in', param='x_pu')
        a = utils.connectivity_mat(self.pds.lines, from_col='from_bus', to_col='to_bus')

        power_mat = self.wds.combs[[_ for _ in self.wds.combs.columns if _.startswith('power')]].fillna(0).values.T
        power_mat = (power_mat * 1000) / (self.pds.power_base_mva * 10 ** 6)
        if x_pumps is not None:
            pumps_power = self.pds.pumps_bus.values @ power_mat @ x_pumps
        else:
            pumps_power = self.pds.pumps_bus.values @ power_mat @ self.x['pumps']

        self.model.st(
            self.pds.gen_mat @ self.x['gen_p']  # generators inflow
            - self.pds.bat_mat @ self.x['bat_p']  # outflow charge batteries - bat_p positive for fill
            + a @ self.x['p'] - r @ self.x['I']  # inflow from lines minus lines losses
            - self.pds.dem_active.values - pumps_power  # outflow demand
            + self.pds.bus.loc[:, 'G'].values @ self.x['v']  # local losses
            + self.x['penalty_p']
            == 0
        )

        self.model.st(self.pds.gen_mat @ self.x['gen_q'] + a @ self.x['q']
                      - x @ self.x['I']
                      - self.pds.dem_reactive.values
                      + self.pds.bus.loc[:, 'B'].values @ self.x['v']
                      + self.x['penalty_q']
                      == 0)

    def penalty_bounds(self, ub):
        self.model.st(self.x['penalty_p'].sum() <= ub)

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
        a = utils.connectivity_mat(self.pds.lines, from_col='from_bus', to_col='to_bus')
        self.model.st(a.T @ self.x['v']
                      + 2 * ((self.x['p'].T * r).T + (self.x['q'].T * x).T)
                      - (self.x['I'].T * (r ** 2 + x ** 2)).T
                      == 0)

    def disable_power_line(self, line_idx):
        self.model.st(self.x['p'][line_idx, :] == 0)

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
        # self.model.do_math().to_lp('model_scale')
        self.model.solve(grb, display=False, params={"TimeLimit": 500, 'OptimalityTol': 10 ** -8})
        self.objective, self.status = self.model.solution.objval, self.model.solution.status
        if self.status in [gurobipy.gurobipy.GRB.OPTIMAL, gurobipy.gurobipy.GRB.SUBOPTIMAL]:
            wds_cost, pds_cost, generation_cost = self.get_systemwise_costs()
            if self.display:
                print(
                    f'Objective: {self.objective:.2f} | WDS: {wds_cost:.2f} | Generators {generation_cost:.2f}'
                    f' | Grid {pds_cost:.2f}')
                print('======================================================================================')
        elif self.status in [gurobipy.gurobipy.GRB.INFEASIBLE, gurobipy.gurobipy.GRB.INF_OR_UNBD]:
            warnings.warn(f"Solution is INFEASIBLE")
            self.objective = None
            self.status = gurobipy.gurobipy.GRB.INFEASIBLE
        else:
            warnings.warn(f"Solution status warning: {self.status} --> , {utils.GRB_STATUS[self.status]}")

    def get_systemwise_costs(self):
        wds_power = self.wds.combs.loc[:, "total_power"].values.reshape(1, -1) @ self.x['pumps'].get()
        wds_cost = (self.wds.tariffs.values.T * wds_power).sum()
        grid_cost = np.sum(self.pds.gen_mat @ (self.pds.pu_to_kw * self.x['gen_p'].get()) * self.pds.tariffs.values)

        # NOTE THAT C IS MULTIPLIED BY T - NUMBER OF TIME STEPS
        # ASSUMING GENERATORS ARE NOT TURNED OFF - THUS FIXED COST IS FOR EVERY TIME STEP
        generated_kw = self.x['gen_p'].get() * self.pds.pu_to_kw
        generation_cost = (np.sum((((self.pds.bus['a'].values.reshape(-1, 1) ** 0.5) * generated_kw).flatten()) ** 2)
                           + (self.pds.bus['b'].values.reshape(-1, 1) * generated_kw).sum()
                           + (self.pds.bus['c'].values.reshape(-1, 1) * self.t).sum()
                           )
        return wds_cost, grid_cost, generation_cost


def solve_water(pds_dir, wds_dir):
    model = Optimizer(pds_data=os.path.join('data', pds_dir), wds_data=os.path.join('data', wds_dir), t=24)
    model.build_water_problem()
    model.solve()
    return model


def solve_combined(pds_dir, wds_dir, x_pumps=None):
    model = Optimizer(pds_data=os.path.join('data', pds_dir), wds_data=os.path.join('data', wds_dir), t=24)
    model.build_combined_problem(x_pumps)
    model.solve()
    return model
