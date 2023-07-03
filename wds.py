import os
import pandas as pd
import numpy as np
import yaml
import bisect
from scipy.interpolate import UnivariateSpline

import utils

pipes_max_flow = {100: 40, 150: 100, 200: 125, 250: 250, 300: 400, 330: 525, 480: 1450, 680: 1750}

class WDS:
    def __init__(self, data_folder):
        self.data_folder = data_folder

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

        self.get_pipes_resistance()
        self.convert_to_cmh()
        self.pipes_pl = self.linearize_pipes()

    def convert_to_cmh(self):
        self.demands *= 3600

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

    @staticmethod
    def pipes_piecewise_linear(n, d, r):
        # find diameter to set max flow
        keys = sorted(pipes_max_flow.keys())
        idx = bisect.bisect_right(keys, d)
        next_diam = keys[idx]

        max_flow = pipes_max_flow[next_diam]

        x = np.linspace(0, max_flow, n)
        y = r * (x ** 1.852)

        f = UnivariateSpline(x, y, k=1, s=0)
        pl = {}
        for i in range(len(x)-1):
            a, b = utils.linear_coefficients_from_two_pints((x[i], float(f(x[i]))), (x[i+1], float(f(x[i+1]))))
            pl[i] = {'start': (x[i], float(f(x[i]))), 'end': (x[i+1], float(f(x[i+1]))), 'a': a, 'b': b}

        return pl

    def linearize_pipes(self):
        pipes_pl = {}
        for pipe_id, row in self.pipes.iterrows():
            pipes_pl[pipe_id] = self.pipes_piecewise_linear(n=4, d=row['diameter_m']*1000, r=row['R'])

        return pipes_pl

    def tank_vol_to_level(self, tank_idx, vol):
        diameter = self.tanks.loc[tank_idx, 'diameter']
        return (4 * vol) / (np.pi * diameter ** 2)