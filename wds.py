import os
import pandas as pd
import numpy as np
import yaml


class WDS:
    def __init__(self, data_folder):
        self.data_folder = data_folder

        self.pumps = pd.read_csv(os.path.join(self.data_folder, 'pumps.csv'), index_col=0)

        self.n_pumps = len(self.pumps)


