import numpy as np
import rsome
from rsome import ro
from rsome import grb_solver as grb

import graphs
import utils
from pds import PDS
from wds import WDS


class Opt:
    def __init__(self, pds_data: str, wds_data: str, t: int, n: int):
        self.pds_data = pds_data
        self.wds_data = wds_data
        self.t = t
        self.n = n  # number of breakpoints for piecewise linearization

        self.pds, self.wds = self.init_distribution_systems()
        self.model = ro.Model()
        self.x = self.declare_vars()
        self.pl_flow_mat = self.build_piecewise_matrices('flow')
        self.pl_head_mat = self.build_piecewise_matrices('head')
        self.pl_power_mat = self.build_piecewise_matrices('power')
        self.build_opt_problem()

    def init_distribution_systems(self):
        pds = PDS(self.pds_data)
        wds = WDS(self.wds_data, self.n)
        return pds, wds

    def declare_vars(self):
        gen_p = self.model.dvar((self.pds.n_bus, self.t))  # active power generation
        gen_q = self.model.dvar((self.pds.n_bus, self.t))  # reactive power generation
        psh_y = self.model.dvar((self.pds.n_psh, self.t))  # psh_y = pumped storage hydropower injection
        psh_h = self.model.dvar((self.pds.n_psh, self.t))  # psh_y = pumped storage hydropower consumption

        v = self.model.dvar((self.pds.n_bus, self.t))  # buses squared voltage
        I = self.model.dvar((self.pds.n_lines, self.t))  # lines squared current flow
        p = self.model.dvar((self.pds.n_lines, self.t))  # lines active power flow
        q = self.model.dvar((self.pds.n_lines, self.t))  # lines reactive power flow
        #
        self.model.st(gen_p >= 0)
        self.model.st(gen_q >= 0)
        self.model.st(psh_y >= 0)
        self.model.st(psh_h >= 0)
        self.model.st(I >= 0)

        alpha = self.model.dvar((self.wds.n_pipes, self.t, self.n))  # pipe segments relative flows
        beta = self.model.dvar((self.wds.n_pipes, self.t, self.n - 1), vtype='B')
        h = self.model.dvar((self.wds.n_nodes, self.t))  # nodes head
        pf = self.model.dvar((self.wds.n_pumps, self.t))  # pump flows
        vol = self.model.dvar((self.wds.n_tanks, self.t))  # tanks volume

        self.model.st(alpha <= 1)
        self.model.st(alpha >= 0)
        self.model.st(alpha.sum(axis=-1) == 1)
        self.model.st(beta.sum(axis=-1) == 1)

        # piecewise linear constraints
        mat = np.zeros((self.n - 1, self.n))
        np.fill_diagonal(mat, 1)
        np.fill_diagonal(mat[:, 1:], 1)
        self.model.st(alpha - beta @ mat <= 0)

        return {'gen_p': gen_p, 'gen_q': gen_q, 'psh_y': psh_y, 'psh_h': psh_h, 'v': v, 'I': I, 'p': p, 'q': q,
                'vol': vol, 'alpha': alpha, 'h': h, 'pf': pf, 'beta': beta}

    def build_opt_problem(self):
        self.objective_func()
        self.bus_balance()
        self.energy_conservation()
        self.voltage_bounds()
        self.power_flow_constraint()
        self.water_mass_balance()
        self.no_pumps_backflow()
        self.head_boundaries()
        self.head_conservation()

    def build_piecewise_matrices(self, param):
        """ Build matrices of the x, and y values at the piecewise linear breakpoints """
        mat = np.zeros((self.wds.n_pipes, self.t, self.n))
        for p in self.wds.pipes.index:
            for point in range(self.n):
                mat[p, :, point] = self.wds.pipes_pl[p][point][param]

        return mat

    def objective_func(self):
        pds_cost = (self.pds.gen_mat @ (self.pds.pu_to_kw * self.x['gen_p']) @ self.pds.tariffs.values).sum()
        pumps = self.pumps_power().sum(axis=0)
        if self.wds.n_turbines > 0:
            turbine = self.turbine_power().sum(axis=0)
        else:
            turbine = 0

        wds_cost = (pumps - 0.9 * turbine) @ self.wds.tariffs.sum(axis=1).values
        # psh_cost = (self.pds.psh['fill_tariff'].values @ self.x['psh_y']).sum()
        self.model.min(pds_cost + wds_cost)

    def bus_balance(self):
        r = utils.get_connectivity_mat(self.pds.lines, from_col='from_bus', to_col='to_bus', direction='in',
                                       param='r_pu')
        x = utils.get_connectivity_mat(self.pds.lines, from_col='from_bus', to_col='to_bus', direction='in',
                                       param='x_pu')
        a = utils.get_connectivity_mat(self.pds.lines, from_col='from_bus', to_col='to_bus')

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
        r = self.pds.lines['r_pu'].values.reshape(1, -1)
        x = self.pds.lines['x_pu'].values.reshape(1, -1)
        a = utils.get_connectivity_mat(self.pds.lines, from_col='from_bus', to_col='to_bus')
        self.model.st(a.T @ self.x['v']
                      + 2 * ((self.x['p'].T * r).T + (self.x['q'].T * x).T)
                      - (self.x['I'].T * (r ** 2 + x ** 2)).T
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

    def water_mass_balance(self):
        not_source = utils.get_mat_for_type(self.wds.nodes, 'reservoir', inverse=True)  # all but reservoirs
        a = utils.get_connectivity_mat(self.wds.pipes, from_col='from_node', to_col='to_node')

        tanks_mat = utils.get_mat_for_type(self.wds.nodes, 'tank')
        # rows - all nodes. columns - only tanks
        # tanks_mat @ x_volume results in a matrix with the desired shape
        tanks_mat = tanks_mat[:, ~np.all(tanks_mat == 0, axis=0)]

        dt = utils.get_dt_mat(self.t)
        init_vol = np.zeros((self.wds.n_tanks, self.t))
        init_vol[:, 0] = self.wds.tanks['init_vol'].values
        init_vol = tanks_mat @ init_vol
        self.model.st(not_source @ a @ ((self.pl_flow_mat * self.x['alpha']).sum(axis=-1))
                      - ((tanks_mat @ self.x['vol']) @ dt) + init_vol
                      - self.wds.demands.values == 0)

        self.model.st(self.x['vol'] <= self.wds.tanks['max_vol'].values.reshape(-1, 1))
        self.model.st(self.x['vol'] >= self.wds.tanks['min_vol'].values.reshape(-1, 1))

    def no_pumps_backflow(self):
        pumps_idx = self.wds.pipes.loc[self.wds.pipes['type'] == 'pump'].index
        self.model.st(self.x['alpha'][pumps_idx, :, :] >= 0)

    def head_boundaries(self):
        self.model.st(self.x['h'] >= self.wds.nodes['elevation'].values.reshape(-1, 1))
        tanks_idx = self.wds.nodes.loc[self.wds.nodes['type'] == 'tank'].index.values
        tanks_diam = self.wds.tanks.loc[tanks_idx, 'diameter'].values.reshape(-1, 1)
        self.model.st((self.x['h'])[tanks_idx, :] - (self.x['vol']) * (4 / (np.pi * tanks_diam ** 2)) >= 0)

        res_idx = self.wds.nodes.loc[self.wds.nodes['type'] == 'reservoir'].index
        self.model.st(self.x['h'][res_idx, :] == self.wds.nodes.loc[res_idx, 'elevation'].values.reshape(-1, 1))

    def head_conservation(self):
        """
        numpy allows for "broadcasting" matrix multiplication, to avoid loops
        This will work when the n columns in the first matrix is equal to the n rows in the second matrix.
        numpy has a function called tensordot that can perform the dot product over specified axes for arrays of
        any dimensionality.
        """
        a = utils.get_connectivity_mat(self.wds.pipes, from_col='from_node', to_col='to_node')
        # exclude turbines from headloss constraint
        b = utils.get_mat_for_type(data=self.wds.pipes, element_type='turbine', inverse=True)
        bb = np.tensordot(b, self.pl_head_mat, axes=([1], [0]))

        self.model.st((a @ b).T @ self.x['h'] - ((bb * self.x['alpha']).sum(axis=-1)) == 0)

    def pumps_power(self):
        idx = self.wds.pipes.loc[self.wds.pipes['type'] == 'pump'].index
        pl_power = np.take(self.pl_power_mat, idx, axis=0)
        return (self.pl_power_mat * self.x['alpha']).sum(axis=-1)

    def turbine_power(self):
        idx = self.wds.pipes.loc[self.wds.pipes['type'] == 'turbine'].index
        return (self.pl_power_mat[idx, :, :] * self.x['alpha'][idx, :, :]).sum(axis=-1)

    def solve(self):
        self.model.solve(grb, display=True, params={'TimeLimit': 45, 'MIPGap': 0.01})
        obj, status = self.model.solution.objval, self.model.solution.status
        print(obj, status)

    def extract_res(self, x, elem_type, series_type='time', elem_idx=None, t_idx=None, dec=1, factor=1):
        n = {'nodes': self.pds.n_bus, 'lines': self.pds.n_lines}
        if series_type == 'time':
            values = {t: self.x[x].get()[elem_idx, t] * factor for t in range(self.t)}

        if series_type == 'elements':
            values = {i: self.x[x].get()[i, t_idx] * factor for i in range(n[elem_type])}

        return {t: round(val, dec) for t, val in values.items()}

    def plot_results(self, t):
        n_vals = self.extract_res('v', elem_type='nodes', series_type='elements', t_idx=t, dec=2)
        e_vals = self.extract_res('p', elem_type='lines', series_type='elements', t_idx=t, factor=self.pds.pu_to_kw)
        gr = graphs.OptGraphs(self, self.pds, self.wds)
        gr.plot_graph(self.pds.lines, coords=self.pds.coords, from_col='from_bus', to_col='to_bus',
                      edges_values=e_vals, nodes_values=n_vals)

        gr.plot_graph(self.wds.pipes, coords=self.wds.coords, from_col='from_node', to_col='to_node',
                      edges_values={i: (self.x['alpha'].get() * self.pl_flow_mat).sum(axis=-1)[i, t]

                                    for i in range(self.wds.n_pipes)},
                      nodes_values={i: round(self.x['h'].get()[i, t], 1) for i in range(self.wds.n_nodes)}
                      )

        gr.bus_voltage(t=0)
        gr.plot_all_tanks()
        gr.plot_all_pumps()
