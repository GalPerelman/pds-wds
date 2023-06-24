import os
import pandas as pd
import numpy as np
import yaml


class PDS:
    def __init__(self, data_folder):
        self.data_folder = data_folder

        self.bus = pd.read_csv(os.path.join(self.data_folder, 'bus.csv'), index_col=0)
        self.lines = pd.read_csv(os.path.join(self.data_folder, 'lines.csv'), index_col=0)
        self.psh = pd.read_csv(os.path.join(self.data_folder, 'psh.csv'), index_col=0)
        self.dem_active = pd.read_csv(os.path.join(self.data_folder, 'dem_active_power.csv'), index_col=0)
        self.dem_reactive = pd.read_csv(os.path.join(self.data_folder, 'dem_reactive_power.csv'), index_col=0)
        self.grid_tariff = pd.read_csv(os.path.join(self.data_folder, 'grid_tariff.csv'), index_col=0)

        self.n_bus = len(self.bus)
        self.n_lines = len(self.lines)
        self.n_psh = len(self.psh)

        self.convert_resistance_units()
        self.gen_idx = np.where(self.bus['type'] == 'gen', 1, 0)
        self.gen_mat = self.gen_idx * np.eye(self.n_bus)
        self.A = self.get_connectivity_mat()

        # read other parameters
        with open(os.path.join(self.data_folder, 'params.yaml'), 'r') as f:
            params = yaml.safe_load(f)
            self.__dict__.update(params)

    def convert_resistance_units(self):
        self.lines['r_kohm'] = self.lines['r_ohm'] / 1000
        self.lines['x_kohm'] = self.lines['x_ohm'] / 1000

    def get_bus_lines(self, bus_id):
        return self.lines.loc[(self.lines['from_bus'] == bus_id) | (self.lines['to_bus'] == bus_id)]

    def get_connectivity_mat(self, param=''):
        mat = np.zeros((self.n_bus, self.n_bus))

        start_indices = np.searchsorted(self.bus.index, self.lines.loc[:, 'from_bus'])
        end_indices = np.searchsorted(self.bus.index, self.lines.loc[:, 'to_bus'])
        if param:
            mat_values = self.lines[param]
        else:
            mat_values = 1

        mat[start_indices, end_indices] = mat_values
        return mat

    def bus_lines_mat(self, direction='both', param=''):
        mat = np.zeros((self.n_bus, self.n_lines))
        mat[self.lines.loc[:, 'from_bus'], np.arange(self.n_lines)] = -1
        mat[self.lines.loc[:, 'to_bus'], np.arange(self.n_lines)] = 1

        if direction == 'in':
            mat[mat == -1] = 0
        if direction == 'out':
            mat[mat == 1] = 0

        if param:
            # row-wise multiplication
            mat = mat * self.lines[param].values
        return mat