import os

import rsome
from rsome import ro
from rsome import grb_solver as grb
import matplotlib.pyplot as plt

import graphs
import utils
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
        gen_p = self.model.dvar((self.pds.n_bus, self.T))  # active power generation
        gen_q = self.model.dvar((self.pds.n_bus, self.T))  # reactive power generation
        psh_y = self.model.dvar((self.pds.n_psh, self.T))  # psh_y = pumped storage hydropower injection
        psh_h = self.model.dvar((self.pds.n_psh, self.T))  # psh_y = pumped storage hydropower consumption

        v = self.model.dvar((self.pds.n_bus, self.T))  # buses squared voltage
        I = self.model.dvar((self.pds.n_lines, self.T))  # lines squared current flow
        p = self.model.dvar((self.pds.n_lines, self.T))  # lines active power flow
        q = self.model.dvar((self.pds.n_lines, self.T))  # lines reactive power flow

        self.model.st(gen_p >= 0)
        self.model.st(gen_q >= 0)
        self.model.st(psh_y >= 0)
        self.model.st(psh_h >= 0)
        self.model.st(I >= 0)

        f = self.model.dvar((self.wds.n_pipes, self.T))  # pipe flows
        h = self.model.dvar((self.wds.n_nodes, self.T))  # nodes head
        pf = self.model.dvar((self.wds.n_pumps, self.T))  # pump flows

        return {'gen_p': gen_p, 'gen_q': gen_q, 'psh_y': psh_y, 'psh_h': psh_h, 'v': v, 'I': I, 'p': p, 'q': q,
                'f': f, 'h': h, 'pf': pf}

    def build(self):
        self.objective_func()
        self.bus_balance()
        self.energy_conservation()
        self.voltage_bounds()
        self.power_flow_constraint()

    def objective_func(self):
        pds_cost = (self.pds.gen_mat @ (self.pds.pu_to_kw * self.x['gen_p']) @ self.pds.grid_tariff.values).sum()
        wds_cost = 0
        psh_cost = (self.pds.psh['fill_tariff'].values @ self.x['psh_y']).sum()
        self.model.min(pds_cost + wds_cost + psh_cost)

    def bus_balance(self):
        r = utils.get_connectivity_mat(self.pds.lines, from_col='from_bus', to_col='to_bus', direction='in', param='r_pu')
        x = utils.get_connectivity_mat(self.pds.lines, from_col='from_bus', to_col='to_bus', direction='in', param='x_pu')
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
        for t in range(self.T):
            for line in range(self.pds.n_lines):
                b_id = self.pds.lines.loc[line, 'to_bus']
                self.model.st(rsome.rsocone(self.x['p'][line, t] + self.x['q'][line, t],
                                            self.x['v'][b_id, t],
                                            self.x['I'][line, t]))

    def water_mass_balance(self):
        pass

    def solve(self):
        self.model.solve(grb, display=True)
        obj, status = self.model.solution.objval, self.model.solution.status
        print(obj, status)

    def extract_res(self, x, elem_type, series_type='time', elem_idx=None, t_idx=None, dec=1, factor=1):
        n = {'nodes': self.pds.n_bus, 'lines': self.pds.n_lines}
        if series_type == 'time':
            values = {t: self.x[x].get()[elem_idx, t] * factor for t in range(self.T)}

        if series_type == 'elements':
            values = {i: self.x[x].get()[i, t_idx] * factor for i in range(n[elem_type])}

        return {t: round(val, dec) for t, val in values.items()}

    def plot_results(self, t, net_coords):
        n_vals = self.extract_res('v', elem_type='nodes', series_type='elements', t_idx=0, dec=2)
        e_vals = self.extract_res('p', elem_type='lines', series_type='elements', t_idx=0, factor=self.pds.pu_to_kw)
        gr = graphs.OptGraphs(self.pds, self.x)
        gr.pds_graph(edges_values=e_vals, nodes_values=n_vals, net_coords=net_coords)
        gr.bus_voltage(t=0)
        graphs.time_series(x=self.pds.dem_active.columns, y=self.x['gen_p'].get()[t, :] * self.pds.pu_to_kw)


if __name__ == "__main__":
    PDS_DATA = os.path.join('data', 'pds')
    WDS_DATA = os.path.join('data', 'wds')
    opt = Opt(pds_data=PDS_DATA, wds_data=WDS_DATA, T=24)
    opt.solve()
    opt.plot_results(t=0, net_coords=graphs.IEEE33_POS)

    plt.show()
