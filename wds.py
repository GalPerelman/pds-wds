import os
import pandas as pd
import numpy as np
import yaml
import bisect
from scipy.interpolate import UnivariateSpline

import utils

pipes_max_flow = {100: 40, 150: 100, 200: 125, 250: 250, 300: 400, 330: 525, 480: 1450, 680: 1750}


class WDS:
    def __init__(self, data_folder, n):
        self.data_folder = data_folder
        self.n = n  # number of piecewise linear segments

        self.nodes = pd.read_csv(os.path.join(self.data_folder, 'nodes.csv'), index_col=0)
        self.pipes = pd.read_csv(os.path.join(self.data_folder, 'pipes.csv'), index_col=0)
        self.pumps = pd.read_csv(os.path.join(self.data_folder, 'pumps.csv'), index_col=0)
        self.tanks = pd.read_csv(os.path.join(self.data_folder, 'tanks.csv'), index_col=0)
        self.demands = pd.read_csv(os.path.join(self.data_folder, 'demands.csv'), index_col=0).T  # index=nodes
        self.tariffs = pd.read_csv(os.path.join(self.data_folder, 'tariffs.csv'), index_col=0)

        # read other parameters
        with open(os.path.join(self.data_folder, 'params.yaml'), 'r') as f:
            params = yaml.safe_load(f)
            self.__dict__.update(params)

        self.n_nodes = len(self.nodes)
        self.n_pipes = len(self.pipes)
        self.n_pumps = len(self.pumps)
        self.n_tanks = len(self.tanks)
        self.n_turbines = len(self.pipes.loc[self.pipes['type'] == 'turbine'])

        self.get_pipes_resistance()
        self.convert_to_cmh()
        self.pipes_pl = self.piecewise_linear()

    def convert_to_cmh(self):
        self.demands *= 3600
        self.pipes['min_flow_cmh'] = self.pipes['min_flow_cms'] * 3600
        self.pipes['max_flow_cmh'] = self.pipes['max_flow_cms'] * 3600
        self.pumps['b'] /= (3600 ** 2)

    @staticmethod
    def hazen_wiliams(c, d, l):
        """
        c: roughness: (dimensionless)
        d: diamater: (m)
        l: length: (m)
        return: head loss (m) coefficient
        """
        return (1.1311 * (10 ** 9)) * ((1 / c) ** 1.852) * ((1 / (d * 1000)) ** 4.87) * l

    def get_pipes_resistance(self):
        """ Calculate pipes resistance coefficients
            Head loss along pipes is the coefficient multiplied by: Q ^ 1.852
            Where Q is the pipe flow in cubic meter per hour
        """
        self.pipes['R'] = self.hazen_wiliams(self.pipes['c'], self.pipes['diameter_m'], self.pipes['length_m'])

    def get_pump_head(self, pump_id, flow):
        a = - (self.pumps.loc[pump_id, 'h_nom']) / (3 * (self.pumps.loc[pump_id, 'q_nom']) ** 2)
        h = a * (flow ** 2) + (4 / 3) * (self.pumps.loc[pump_id, 'h_nom'])
        return h

    def get_pump_power(self, pump_id, flow):
        h = self.get_pump_head(pump_id, flow)

        # build efficiency curve
        a = - (self.pumps.loc[pump_id, 'e_nom']) / (self.pumps.loc[pump_id, 'q_nom'] ** 2)
        b = 2 * (self.pumps.loc[pump_id, 'e_nom']) / (self.pumps.loc[pump_id, 'q_nom'])
        eff = a * (flow ** 2) + b * flow

        power = np.divide((9810 * (flow / 3600) * h), (1000 * eff),
                          out=np.zeros_like((9810 * (flow / 3600) * h)), where=(1000 * eff) != 0)

        return power

    def piecewise_linear(self):
        pipes_pl = {}
        for pipe_id, row in self.pipes.iterrows():
            if row['type'] == 'pipe':
                x = np.linspace(-row['max_flow_cmh'], row['max_flow_cmh'], self.n)
                h = -row['R'] * x * (np.abs(x)) ** 0.852
                p = 0 * x
            if row['type'] == 'pump' or row['type'] == 'turbine':
                x = np.linspace(0,  row['max_flow_cmh'], self.n)
                h = self.get_pump_head(pipe_id, x)
                p = self.get_pump_power(pipe_id, x)

            head_points = UnivariateSpline(x, h, k=1, s=0)
            power_points = UnivariateSpline(x, p, k=1, s=0)
            pl = {}
            for i in range(len(x)):
                pl[i] = {'flow': x[i], 'head': float(head_points(x[i])), 'power': float(power_points(x[i]))}

            pipes_pl[pipe_id] = pl

        return pipes_pl

    def tank_vol_to_level(self, tank_idx, vol):
        diameter = self.tanks.loc[tank_idx, 'diameter']
        return (4 * vol) / (np.pi * diameter ** 2)