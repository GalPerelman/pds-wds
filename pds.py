import os
import pandas as pd
import numpy as np
import yaml

import utils


class PDS:
    def __init__(self, data_folder: str):
        """
        The units of the input data files:
        bus             Vmin_pu:            pu
        bus             Vmax_pu:            pu
        lines           r_ohm:              ohm
        lines           x_ohm:              ohm
        active_dem:     all                 kW
        reactive_dem:   all                 kW
        tariffs:        all                 S/kWh
        generators      min_gen_kw:         kW
        generators      max_gen_kw:         kW
        generators      a:                  $/(kW^2)h
        generators      b:                  $/kWh
        generators      c:                  $

        """
        self.data_folder = data_folder

        self.bus = pd.read_csv(os.path.join(self.data_folder, 'bus.csv'), index_col=0)
        self.lines = pd.read_csv(os.path.join(self.data_folder, 'lines.csv'), index_col=0)
        self.psh = pd.read_csv(os.path.join(self.data_folder, 'psh.csv'), index_col=0)
        self.dem_active = pd.read_csv(os.path.join(self.data_folder, 'dem_active_power.csv'), index_col=0)  # kW
        self.dem_reactive = pd.read_csv(os.path.join(self.data_folder, 'dem_reactive_power.csv'), index_col=0)  # kW
        self.tariffs = pd.read_csv(os.path.join(self.data_folder, 'tariffs.csv'), index_col=0)
        self.generators = pd.read_csv(os.path.join(self.data_folder, 'generators.csv'), index_col=0)
        self.max_gen_profile = pd.read_csv(os.path.join(self.data_folder, 'max_gen_profile.csv'), index_col=0)  # MW

        # read other parameters
        with open(os.path.join(self.data_folder, 'params.yaml'), 'r') as f:
            params = yaml.safe_load(f)
            self.__dict__.update(params)

        self.n_bus = len(self.bus)
        self.n_lines = len(self.lines)
        self.n_psh = len(self.psh)
        self.n_generators = len(self.generators)

        self.factorize_demands()
        self.gen_mat = utils.get_mat_for_type(self.bus, "gen")
        self.construct_generators_params()
        self.pu_to_kw, self.pu_to_kv = self.convert_to_pu()

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
        """
        Function for converting input data to PU (dimensionless per unit)
        The conversion is done by remove all magnitude units (Kilo, Mega etc.) and divide by base values
        base voltage and base power - defined in the input params.yaml file
        return:
            values to inverse conversion back to physical units
            pu_to_kw, pu_to_kv
        """
        z = ((self.nominal_voltage_kv * 1000) ** 2) / (self.power_base_mva * 10 ** 6)
        self.lines['r_pu'] = self.lines['r_ohm'] / z
        self.lines['x_pu'] = self.lines['x_ohm'] / z
        self.dem_active = (self.dem_active * 1000) / (self.power_base_mva * 10 ** 6)
        self.dem_reactive = (self.dem_reactive * 1000) / (self.power_base_mva * 10 ** 6)
        self.bus['max_gen_kw'] = (self.bus['max_gen_kw'] * 1000) / (self.power_base_mva * 10 ** 6)
        self.bus['min_gen_kw'] = (self.bus['min_gen_kw'] * 1000) / (self.power_base_mva * 10 ** 6)
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

    def construct_bus_pumps_mat(self):
        mat = np.zeros((len(self.bus), int(self.bus['pump_id'].max() + 1)))

        # Get row and column indices where the bus are connected to pumps (pump_id col is not Nan)
        row_indices = self.bus.index[~self.bus['pump_id'].isna()].to_numpy()
        col_indices = self.bus['pump_id'].dropna().astype(int).to_numpy()

        # Use row and column indices to set corresponding elements in the matrix to 1
        mat[row_indices, col_indices] = 1
        return mat

    def construct_generators_params(self):
        self.bus = pd.merge(self.bus, self.generators, left_index=True, right_index=True, how='outer')
        self.bus[['a', 'b', 'c']] = self.bus[['a', 'b', 'c']].fillna(0)
        self.bus['min_gen_kw'] = self.bus[['min_gen_kw']].fillna(0)
        self.bus['max_gen_kw'] = self.bus[['max_gen_kw']].fillna(np.inf)
