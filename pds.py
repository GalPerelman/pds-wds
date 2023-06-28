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

        # read other parameters
        with open(os.path.join(self.data_folder, 'params.yaml'), 'r') as f:
            params = yaml.safe_load(f)
            self.__dict__.update(params)

        self.n_bus = len(self.bus)
        self.n_lines = len(self.lines)
        self.n_psh = len(self.psh)

        self.pu_to_kw, self.pu_to_kv = self.convert_to_pu()
        self.factorize_demands()

        self.gen_idx = np.where(self.bus['type'] == 'gen', 1, 0)
        self.gen_mat = self.gen_idx * np.eye(self.n_bus)
        self.A = self.get_connectivity_mat()

    def factorize_demands(self):
        try:
            self.dem_active *= self.active_demand_factor
        except AttributeError:
            pass

        try:
            self.dem_reactive *= self.reactive_demand_factor
        except AttributeError:
            pass

    def convert_to_pu(self):
        z = ((self.nominal_voltage_kv * 1000) ** 2) / (self.power_base_mva * 10 ** 6)
        self.lines['r_pu'] = self.lines['r_ohm'] / z
        self.lines['x_pu'] = self.lines['x_ohm'] / z
        self.dem_active = (self.dem_active * 1000) / (self.power_base_mva * 10 ** 6)
        self.dem_reactive = (self.dem_reactive * 1000) / (self.power_base_mva * 10 ** 6)
        return self.power_base_mva * 10 ** 6 / 1000, z ** (-1)

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
    