import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
import matplotlib.patheffects as PathEffects


class OptGraphs:
    def __init__(self, opt):
        self.opt = opt
        self.pds = opt.pds
        self.wds = opt.wds
        self.x = self.opt.x

    def plot_graph(self, edges_data, coords, from_col, to_col, edges_values={}, nodes_values={}, title=''):
        df = edges_data.copy()
        if edges_values:
            df['w'] = edges_values.values()
        else:
            df['w'] = 1
        G = nx.from_pandas_edgelist(df, source=from_col, target=to_col, edge_attr='w')

        nx.set_node_attributes(G, coords, 'coord')
        nx.set_node_attributes(G, nodes_values, 'size')
        options = {
            "font_size": 9,
            "node_size": 200,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 1,
            "width": 1,
        }
        fig, ax = plt.subplots(figsize=(12, 5))
        nx.draw_networkx(G, pos=coords, ax=ax, **options)

        if nodes_values:
            labels_pos = {k: (v[0] + 0.2, v[1] + 0.15) for k, v in coords.items()}
            nx.draw_networkx_labels(G, pos=labels_pos, ax=ax, labels=nodes_values, font_size=8, font_color="blue")

        if edges_values:
            edge_labels = {k: round(v, 1) for k, v in nx.get_edge_attributes(G, 'w').items()}
            nx.draw_networkx_edge_labels(G, coords, ax=ax, edge_labels=edge_labels, font_size=8)

        plt.subplots_adjust(bottom=0.04, top=0.96, left=0.04, right=0.96)
        plt.suptitle(title)
        return fig

    def bus_voltage(self, t, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar(range(self.pds.n_bus), self.x['v'].get()[:, t])
        return fig

    def plot_all_tanks(self, fig=None, leg_label=None):
        n = self.wds.n_tanks
        ncols = max(1, int(math.ceil(math.sqrt(n))))
        nrows = max(1, int(math.ceil(n / ncols)))

        if fig is None:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(8, 4))
            axes = np.atleast_2d(axes).ravel()
        else:
            axes = fig.axes

        for i, (tank_id, row) in enumerate(self.wds.tanks.iterrows()):
            values = [row['init_vol']] + list(self.x['vol'].get()[i, :])
            title = f'Tank {tank_id}'
            axes[i] = time_series(x=range(len(values)), y=values,
                                  title=title, ax=axes[i], ylabel=f'Volume ($m^3)$', leg_label=leg_label)
            axes[i].legend()

        plt.tight_layout()
        return fig

    def plot_all_pumps(self):
        n = self.wds.n_pumps
        ncols = max(1, int(math.ceil(math.sqrt(n))))
        nrows = max(1, int(math.ceil(n / ncols)))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(8, 4))
        axes = np.atleast_2d(axes).ravel()
        for i, (pump_id, row) in enumerate(self.wds.pumps.iterrows()):
            values = (self.x['alpha'].get() * self.opt.pl_flow_mat).sum(axis=-1)[pump_id, :]
            axes[i] = time_series(x=range(len(values)), y=values, title=f'Pump {pump_id}', ax=axes[i])

        plt.tight_layout()
        return fig

    def pumps_gantt(self, pumps_names: list):
        df = pd.DataFrame(self.x['pumps'].get()).T

        pe = [PathEffects.Stroke(linewidth=6, foreground='black'),
              PathEffects.Stroke(foreground='black'),
              PathEffects.Normal()]

        fig, ax = plt.subplots()
        for i, unit in enumerate(pumps_names):
            unit_combs = self.wds.combs.loc[self.wds.combs[unit] == 1].index.to_list()
            temp = pd.DataFrame((df[unit_combs] > 0).any(axis=1).astype(int), columns=[unit], index=df.index)
            temp.loc[:, 'start'] = temp.index
            temp.loc[:, 'end'] = temp['start'] + pd.Series(temp[unit])
            ax.hlines([i], temp.index.min(), temp.index.max() + 1, linewidth=5, color='w', path_effects=pe)
            ax.hlines(np.repeat(i, len(temp)), temp['end'], temp['start'], linewidth=5, colors='black')

        ax.xaxis.grid(True)
        ax.set_yticks([i for i in range(len(pumps_names))])
        ax.set_yticklabels(pumps_names)
        return fig

    def plot_all_generators(self, fig=None, leg_label=''):
        gen_idx = self.pds.bus.loc[self.pds.bus['type'] == 'gen'].index
        x = self.x['gen_p'].get()[gen_idx, :] * self.pds.pu_to_kw
        x = x[~np.all(np.isclose(x, 0, atol=0.0001), axis=1)]

        ncols = max(1, int(math.ceil(math.sqrt(x.shape[0]))))
        nrows = max(1, int(math.ceil(x.shape[0] / ncols)))
        if fig is None:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(8, 4))
            axes = np.atleast_2d(axes).ravel()
        else:
            axes = fig.axes

        for i in range(x.shape[0]):
            axes[i] = time_series(range(x.shape[1]), x[i, :], ax=axes[i], ylabel='Power (kW)', leg_label=leg_label)
            axes[i].legend()

        return fig

    def plot_batteries(self, fig=None, leg_label=''):
        bat_idx = self.pds.batteries.index.tolist()
        x = self.x['bat_e'].get()[bat_idx, :] * self.pds.pu_to_kw
        x = x[~np.all(np.isclose(x, 0, atol=0.0001), axis=1)]

        ncols = max(1, int(math.ceil(math.sqrt(x.shape[0]))))
        nrows = max(1, int(math.ceil(x.shape[0] / ncols)))
        if fig is None:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(8, 4))
            axes = np.atleast_2d(axes).ravel()

        else:
            axes = fig.axes

        for i in range(x.shape[0]):
            axes[i] = time_series(range(x.shape[1]), x[i, :], ax=axes[i], ylabel='Energy (kWh)', leg_label=leg_label)
            axes[i].legend()

        return fig


def plot_demands(pds, wds):
    fig, axes = plt.subplots(nrows=2, sharex=True)

    axes[0].plot(pds.dem_active.sum(axis=0).T * pds.pu_to_kw, marker='o', markersize=4, mfc='w')
    axes[0].grid()
    axes[0].set_ylabel('Active power (kW)')

    axes[1].plot(wds.demands.sum(axis=1).index, wds.demands.sum(axis=1), marker='o', markersize=4, mfc='w')
    axes[1].grid()
    axes[1].set_ylabel('Water demand (CMH)')

    fig.align_ylabels()
    return fig


def time_series(x, y, ax=None, ylabel='', title='', leg_label=''):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(x, y, label=leg_label)
    ax.grid(True)

    if ylabel:
        ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    return ax
