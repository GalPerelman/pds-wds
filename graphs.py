import matplotlib.pyplot as plt
import networkx as nx


def plot_graph(df, weights=None):
    G = nx.from_pandas_edgelist(df, source='from_bus', target='to_bus', edge_attr=weights)

    pos = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0), 4: (4, 0), 5: (5, 0), 6: (6, 0), 7: (7, 0),
           8: (8, 0), 9: (9, 0), 10: (10, 0), 11: (11, 0), 12: (12, 0), 13: (13, 0), 14: (14, 0), 15: (15, 0),
           16: (16, 0), 17: (17, 0),
           18: (1, -1), 19: (2, -1), 20: (3, -1), 21: (4, -1),
           22: (2, 2), 23: (3, 2), 24: (4, 2),
           25: (5, 1), 26: (6, 1), 27: (7, 1), 28: (8, 1), 29: (9, 1), 30: (10, 1), 31: (11, 1), 32: (12, 1)
           }

    nx.set_node_attributes(G, pos, 'coord')
    options = {
        "font_size": 9,
        "node_size": 200,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 1,
        "width": 1,
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    nx.draw_networkx(G, pos=pos, ax=ax, **options)
    labels = {k: round(v, 1) for k, v in nx.get_edge_attributes(G, weights).items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=8)
    plt.subplots_adjust(bottom=0.04, top=0.96, left=0.04, right=0.96)
    return G
