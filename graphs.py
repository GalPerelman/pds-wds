import matplotlib.pyplot as plt
import networkx as nx


class OptGraphs:
    def __init__(self, pds, x):
        self.pds = pds
        self.x = x

    def plot_graph(self, edges_data, coords, from_col, to_col, edges_values={}, nodes_values={}):
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

    def bus_voltage(self, t, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar(range(self.pds.n_bus), self.x['v'].get()[:, t])


def time_series(x, y, ax=None, ylabel=''):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(x, y)
    ax.grid()

    if ylabel:
        ax.set_ylabel(ylabel)
