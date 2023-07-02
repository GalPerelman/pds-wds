import os
import pandas as pd
import numpy as np
import yaml


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
        return (1.1311 * 10 ** 9) * ((1 / c) ** 1.852) * ((d * 1000) ** -4.87) * l

    def get_pipes_resistance(self):
        """ Calculate pipes resistance coefficients
            Head loss along pipes is the coefficient multiplied by: Q ^ 1.852
            Where Q is the pipe flow in cubic meter per hour
        """
        self.pipes['R'] = self.hazen_wiliams(self.pipes['c'], self.pipes['diameter_m'], self.pipes['length_m'])
