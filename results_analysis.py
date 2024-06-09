import math

import numpy as np
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
import scipy
from matplotlib import cm
from scipy.interpolate import griddata
import seaborn as sns
import plotly.express as px

from pds import PDS
from wds import WDS

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

COLORS = ["#253494", "#59A9FF", "#de2d26"]


def load_results(results_file_path: str, drop_nans=True):
    data = pd.read_csv(results_file_path, index_col=0)
    if drop_nans:
        data = data.dropna(axis=0, how='any')

    data["tanks_state"] = data["tanks_state"].apply(literal_eval)
    data["batteries_state"] = data["batteries_state"].apply(literal_eval)
    data["outage_lines"] = data["outage_lines"].apply(literal_eval)

    data["coord_dist_reduction"] = (100 * (data["coordinated_distributed"] - data["decentralized"]) /
                                    data["decentralized"])  # "communicate_reduction"
    data["central_reduction"] = (100 * (data["centralized"] - data["decentralized"])
                                 / data["decentralized"])  # cooperative_reduction
    data["coord_dist_diff"] = data["coordinated_distributed"] - data["decentralized"]  # communicate_diff
    data["central_diff"] = data["centralized"] - data["decentralized"]  #

    data['t1_state'] = data['tanks_state'].apply(lambda x: x[0])
    data['tanks_state_avg'] = data['tanks_state'].apply(lambda x: sum(x) / len(x))
    data['batteries_state_avg'] = data['batteries_state'].apply(lambda x: sum(x) / len(x))

    total_load = pds.dem_active.sum() * pds.pu_to_kw
    data["p_ls_decentral"] = data["decentralized"] / total_load  # p_ls_indep
    data["p_ls_coord_dist"] = data["coordinated_distributed"] / total_load  # p_ls_comm
    data["p_ls_central"] = data["centralized"] / total_load  # p_ls_coop

    data.dropna(axis=0, how="any")
    return data


def box(data):
    fig, ax = plt.subplots()
    positions = [0.2, 0.6, 1]
    box_columns = {"decentralized": "Decentralized",
                   "coordinated_distributed": "Coordinated\nDistributed",
                   "centralized": "Centralized"}

    df = pd.DataFrame({"mean": data[box_columns.keys()].mean(),
                       "std": data[box_columns.keys()].std(),
                       "max": data[box_columns.keys()].max(),
                       "min": data[box_columns.keys()].min(),
                       }).T
    df["comm_deviation"] = 100 * (df["coordinated_distributed"] - df["centralized"]) / df["centralized"]
    df["indep_deviation"] = 100 * (df["decentralized"] - df["centralized"]) / df["centralized"]
    print(df)

    ax.boxplot(data[box_columns.keys()],
               labels=box_columns.values(),
               positions=positions, showfliers=False, showmeans=True, patch_artist=True,
               boxprops=dict(facecolor="w"),
               meanprops={'markerfacecolor': 'C0', 'markeredgecolor': "k", "linewidth": 0.1},
               medianprops=dict(linewidth=2, color='C0'))

    for i, col in enumerate(box_columns):
        ax.text(positions[i] + 0.025, data[col].mean() + 500, f"{data[col].mean():.1f}", horizontalalignment='left',
                color='C0', fontsize=9, bbox=dict(facecolor='w', edgecolor="k"))

    ax.set_xlim(0, 1.2)
    ax.set_ylabel("Load Shedding (kWhr)")
    ax.grid()
    plt.subplots_adjust(left=0.15)


def ls_reduction(data, explanatory_var, x_label):
    fig, ax = plt.subplots()
    data = data.sort_values(explanatory_var)
    ax.scatter(data[explanatory_var], data["coord_dist_reduction"], edgecolor="k", linewidth=0.5, s=25, alpha=0.7,
               zorder=2, label="Coordinated Distributed")
    ax.scatter(data[explanatory_var], data["central_reduction"], edgecolor="k", linewidth=0.5, s=25, alpha=0.7,
               zorder=1, label="Centralized")

    ax.grid()
    ax.set_axisbelow(True)
    ax.legend()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    ax.set_xlabel(x_label)
    ax.set_ylabel("LS Reduction (%)")
    plt.subplots_adjust(left=0.15)


def scatter_hist(data, explanatory_var, x_label):
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.15, right=0.95, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    x, y_comm, y_coop = data[explanatory_var], data["coord_dist_reduction"], data["central_reduction"]
    ax.scatter(x, y_comm, edgecolor="k", linewidth=0.5, s=25, alpha=0.7, zorder=2, label="Coord-Distrib")
    ax.scatter(x, y_coop, edgecolor="k", linewidth=0.5, s=25, alpha=0.7, zorder=1, label="Centralized")

    ax_histx.hist(x, bins=20, edgecolor='k', alpha=0.7, zorder=2)
    ax_histy.hist(y_comm, bins=20, orientation='horizontal', edgecolor='k', alpha=0.7, zorder=2)

    ax_histx.hist(x, bins=20, edgecolor='k', alpha=0.7, zorder=1)
    ax_histy.hist(y_coop, bins=20, orientation='horizontal', edgecolor='k', alpha=0.7, zorder=1)

    ax.grid()
    ax.set_axisbelow(True)

    ax.legend()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    ax.set_xlabel(x_label)
    ax.set_ylabel("Load Shedding Reduction (%)")


def analyze_costs(data):
    """
    Assumption: delta_cost = (communicate_cost - independent_cost) + missing_vol_pumping_cost
    Missing volume is the results of canceling the final tanks volume constraint during emergency
    Assumption (1): missing volume will be pumped during On-Peak to catch up with the regular operation
    Assumption (2): power of pumping the missing volume will be the average of all pumping combinations
    """
    data["decentral_specific_cost"] = data["decentralized_wds_cost"] / data["decentralized_wds_vol"]
    data["coordinated_distributed_specific_cost"] = data["coordinated_distributed_wds_cost"] / data[
        "coordinated_distributed_wds_vol"]
    data["specific_cost_increase"] = (data["coordinated_distributed_specific_cost"]) / data["decentral_specific_cost"]

    data["decentralized_final_vol"] = data["decentralized_final_vol"].apply(literal_eval)
    data["coordinated_distributed_final_vol"] = data["coordinated_distributed_final_vol"].apply(literal_eval)

    data["decentralized_total_final_vol"] = data.apply(lambda x: sum(x["decentralized_final_vol"]), axis=1)
    data["coordinated_distributed_total_final_vol"] = data.apply(lambda x: sum(x["coordinated_distributed_final_vol"]),
                                                                 axis=1)
    data["missing_vol"] = data["decentralized_total_final_vol"] - data["coordinated_distributed_total_final_vol"]

    data["missing_cost"] = data["missing_vol"] * 0.44 * 0.25  # vol (m^3) * 0.44 (kWhr / m^3) * 0.25 ($ / kWhr)
    data["delta_cost"] = data["coordinated_distributed_wds_cost"] + data["missing_cost"] - data[
        "decentralized_wds_cost"]
    data["delta_cost_percentage"] = 100 * data["delta_cost"] / data["decentralized_wds_cost"]

    pd.options.display.float_format = '{:.3f}'.format
    x, y = data["coord_dist_reduction"] * -1, data["delta_cost_percentage"]
    fig, ax = plt.subplots()
    ax = sns.regplot(x=x, y=y, color='C0', scatter_kws={"edgecolor": "k", "linewidths": 0.5, "alpha": 0.5, "s": 25},
                     line_kws={"color": "k"}, ax=ax)

    # r, p = scipy.stats.pearsonr(x, y)
    # ax.text(0.05, 0.9, f'$R^2$={r:.2f}', transform=ax.transAxes)

    ax.set_axisbelow(True)
    ax.grid()
    ax.set_xlabel("Load Shedding Reduction (%)")
    ax.set_ylabel("Additional Cost (%)")

    x, y = data["coord_dist_diff"] * -1, data["delta_cost"]
    fig, ax = plt.subplots()
    ax = sns.regplot(x=x, y=y, color='C0', scatter_kws={"edgecolor": "k", "linewidths": 0.5, "alpha": 0.5, "s": 25},
                     line_kws={"color": "k"}, ax=ax)

    ax.set_axisbelow(True)
    ax.grid()
    ax.set_xlabel("Load Shedding Reduction (kWhr)")
    ax.set_ylabel("Additional Cost ($)")

    #  cost for WDS vs VOLL
    # According to https://eta-publications.lbl.gov/sites/default/files/lbnl-6941e.pdf:
    # Cost per not served kwh {duration (hr): $ per unserved kwh, 0.5: 5.9, 1: 3.3, 4: 1.6, 8: 1.4, 16: 1.3}
    # In this study emergency duration ranges from 6-24 hours
    # Accordingly, a linear regression was used to estimate VOLL according to the last 3 points
    # The VOLL is multiplied by the total ls REDUCTION comparing to the independent approach
    # This allow to estimate the contribution of the coordinated distributed approach compared to business as usual
    data = data.sort_values("coord_dist_diff")
    data["voll"] = 0.025 * data["t"] + 1.2
    x, y = data["coord_dist_diff"] * -data["voll"], data["delta_cost"]
    fig, ax = plt.subplots()
    # ax = sns.regplot(x=x, y=y, color='C0', scatter_kws={"edgecolor": "k", "linewidths": 0.5, "alpha": 0.5, "s": 25},
    #                  line_kws={"color": "k"}, ax=ax)
    ax.scatter(x=x, y=y, color='C0', edgecolor="k", linewidths=0.5, alpha=0.5, s=25)
    ax.set_axisbelow(True)
    ax.grid()
    ax.set_xlabel("VOLL Reduction ($)")
    ax.set_ylabel("WDS Additional Cost ($)")


def double_factor_scatter(data, x_col, y_col, z_col):
    x = data[x_col].values
    y = data[y_col].values
    z = data[z_col].values

    grid_x, grid_y = np.mgrid[0:x.max():10j, 0:y.max():10j]
    points = np.array([x, y]).T
    grid = griddata(points, np.array(z), (grid_x, grid_y), method='linear')

    mappable = plt.cm.ScalarMappable(cmap=plt.cm.cividis)
    mappable.set_array(grid)
    mappable.set_clim(z.min(), z.max())

    z2 = data["central_reduction"].values
    grid2 = griddata(points, np.array(z2), (grid_x, grid_y), method='linear')

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_surface(grid_x, grid_y, grid, rstride=1, cstride=1, edgecolor='k', lw=0.5, norm=mappable.norm, alpha=0.8)
    ax.plot_surface(grid_x, grid_y, grid2, rstride=1, cstride=1, edgecolor='k', lw=0.5, norm=mappable.norm, alpha=0.8)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.azim = -140
    ax.elev = 20
    # cax = fig.add_axes([axes[0].get_position().x1 - 0.03, 0.22, 0.02, 0.48])
    # cbar = plt.colorbar(l, cax=cax)
    # cbar.ax.set_title('Total energy\ncosts (€)', fontsize=11)
    # plt.subplots_adjust(left=0.01)
    # plt.subplots_adjust(top=0.972, bottom=0.028, left=0.06, right=0.88, hspace=0.2, wspace=0.35)


def all_factors(data):
    factors_labels = {"t": "Duration", "start_time": "Start Time",
                      "wds_demand_factor": "WDS Demand Multiplier",
                      "pds_demand_factor": "PDS Load Multiplier",
                      "pv_factor": "PV Availability Multiplier",
                      "tanks_state_avg": "Average Tanks Initial State",
                      "batteries_state_avg": "Average Batteries Initial State"}

    ncols = max(1, int(math.ceil(math.sqrt(len(factors_labels))) + 1))
    nrows = max(1, int(math.ceil(len(factors_labels) / ncols)))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=(10, 6))
    axes = axes.ravel()
    for i, (factor, label) in enumerate(factors_labels.items()):
        axes[i].scatter(data[factor], data["coord_dist_reduction"], edgecolor="k", linewidth=0.5, s=25, alpha=0.7,
                        zorder=2)
        axes[i].yaxis.set_major_formatter(mtick.PercentFormatter())
        axes[i].set_xlabel(label)
        axes[i].grid()

    expanded_rows = []
    for index, row in data.iterrows():
        for line in row['outage_lines']:
            expanded_rows.append({'line': line, 'coord_dist_reduction': row['coord_dist_reduction']})

    expanded_df = pd.DataFrame(expanded_rows)
    axes[-1].scatter(expanded_df['line'], expanded_df['coord_dist_reduction'],
                     edgecolor="k", linewidth=0.5, s=25, alpha=0.7, zorder=2)
    axes[-1].set_xlabel("Is line idx outage")
    axes[-1].grid()
    fig.text(0.02, 0.5, 'Coordinated Distributed LS reduction (%)', va='center', rotation='vertical')
    plt.subplots_adjust(top=0.95, bottom=0.11, left=0.12, right=0.97, hspace=0.3, wspace=0.12)


def kde_plot():
    fig, ax = plt.subplots()
    total_loads = pds.dem_active.values.sum() * pds.pu_to_kw
    (100 * data['decentralized'] / total_loads).plot(kind='kde', c=COLORS[2], label="Decentralized", ax=ax)
    (100 * data['coordinated_distributed'] / total_loads).plot(kind='kde', c=COLORS[0], label="Coord-Distrib", ax=ax)
    (100 * data['centralized'] / total_loads).plot(kind='kde', c=COLORS[1], label="Centralized", ax=ax)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlabel("LS out of total loads")
    ax.set_xlim(-3, 40)
    ax.grid()
    ax.legend()


def parallel_coords():
    data_sorted = data.sort_values(by="coord_dist_reduction", ascending=False)
    fig = px.parallel_coordinates(data_sorted
                                  [["coord_dist_reduction",
                                    "t", "start_time", "wds_demand_factor", "pds_demand_factor", "pv_factor",
                                    "tanks_state_avg", "batteries_state_avg"
                                    ]]
                                  ,
                                  color="coord_dist_reduction",
                                  labels={"t": "Duration", "start_time": "Start Time",
                                          "wds_demand_factor": "WDS Demand Multiplier",
                                          "pds_demand_factor": "PDS Load Multiplier",
                                          "pv_factor": "PV Availability Multiplier",
                                          "tanks_state_avg": "Average Tanks Initial State",
                                          "batteries_state_avg": "Average Batteries Initial State"},
                                  color_continuous_scale=px.colors.sequential.Jet,
                                  # color_continuous_midpoint=-8,
                                  # range_color=[-22, 0]
                                  )

    fig.update_traces(dimensions=[
        dict(range=[-22, 0], label="Coordinated Distributed<br>LS Reduction (%)",
             values=data_sorted["coord_dist_reduction"], tickvals=list(np.arange(-22, 2, 2))),
        dict(range=[6, 24], label="Duration", values=data_sorted["t"], tickvals=[_ for _ in range(6, 25)]),
        dict(range=[0, 24], label="Start Time", values=data_sorted["start_time"], tickvals=[_ for _ in range(25)]),
        dict(range=[0.8, 1.2], label="WDS Demand Multiplier", values=data_sorted["wds_demand_factor"],
             tickvals=[0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.2]),
        dict(range=[0.9, 1.2], label="PDS Load Multiplier", values=data_sorted["pds_demand_factor"],
             tickvals=[0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.2]),
        dict(range=[0.8, 1.2], label="PV Availability Multiplier", values=data_sorted["pv_factor"],
             tickvals=[0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.2]),
        dict(range=[0.2, 1], label="Average Tanks<br>Initial State", values=data_sorted["tanks_state_avg"],
             tickvals=[_ / 10 for _ in range(2, 11)]),
        dict(range=[0.2, 1], label="Average Batteries<br>Initial State", values=data_sorted["batteries_state_avg"],
             tickvals=[_ / 10 for _ in range(2, 11)])
    ])

    fig.update_layout(font=dict(family="Arial, sans-serif", size=18, color="black"),
                      coloraxis_colorbar=dict(title="Coordinated Distributed<br>LS Reduction (%)",
                                              titlefont=dict(family="Arial, sans-serif", size=18, color="black"))
                      )
    fig.show()


def mpl_parallel_coordinates(data):
    data["coord_dist_reduction"] *= -1
    _data_sorted = data.sort_values(by="coord_dist_reduction", ascending=True)
    _objs = {"coord_dist_reduction": "Coordinated\nDistributed\nLS Reduction (%)", "t": "Duration",
             "start_time": "Start Time",
             "wds_demand_factor": "WDS Demand\nMultiplier", "pds_demand_factor": "PDS Load\nMultiplier",
             "pv_factor": "PV Availability\nMultiplier", "tanks_state_avg": "Average Tanks\nInitial State (%)",
             "batteries_state_avg": "Average Batteries\nInitial State (%)"}

    objs = list(_objs.values())
    data_sorted = _data_sorted[_objs.keys()].values
    _data_sorted = _data_sorted[_objs.keys()].values
    x = [i for i, _ in enumerate(objs)]
    # sharey=False indicates that all the subplot y-axes will be set to different values
    fig, ax = plt.subplots(1, len(x) - 1, sharey=False, figsize=(12, 5))

    min_max_range = {}
    for i in range(len(objs)):
        data_sorted[:, i] = np.true_divide(data_sorted[:, i] - min(data_sorted[:, i]), np.ptp(data_sorted[:, i]))
        min_max_range[objs[i]] = [min(_data_sorted[:, i]), max(_data_sorted[:, i]), np.ptp(_data_sorted[:, i])]

    colormap = cm.get_cmap('YlGnBu')
    norm = mcolors.Normalize(vmin=0, vmax=1)

    colormap_bad = cm.get_cmap('RdYlBu')
    norm_bad = mcolors.Normalize(vmin=0, vmax=0.4)

    for i, ax_i in enumerate(ax):
        for d in range(len(data_sorted)):
            if data_sorted[d, 0] * min_max_range[objs[0]][2] + min_max_range[objs[0]][0] >= 10:
                color_good = colormap(norm(data_sorted[d, 0]))
                ax_i.plot(objs, data_sorted[d, :], color=color_good, alpha=0.7, linewidth=2, zorder=2)
            # if data_sorted[d, 0] * min_max_range[objs[0]][2] + min_max_range[objs[0]][0] <= 1.5:
            #     color_bad = colormap_bad(norm_bad(data_sorted[d, 0]))
            #     ax_i.plot(objs, data_sorted[d, :], color=color_bad, alpha=0.7, linewidth=1, zorder=1)
            else:
                ax_i.plot(objs, data_sorted[d, :], color='lightgrey', alpha=0.3, linewidth=1, zorder=0)
        ax_i.set_xlim([x[i], x[i + 1]])

    # function for setting ticks and tick_lables along the y-axis of each subplot
    bbox = dict(boxstyle="round", ec="none", fc="white", alpha=0.2)

    def set_ticks_for_axis(dim, ax_i, n_ticks):
        min_val, max_val, v_range = min_max_range[objs[dim]]
        locations = np.linspace(0, 1, n_ticks)
        tick_labels = [round(_ * v_range + min_val, 2) for _ in locations]
        ax_i.yaxis.set_ticks(locations)
        ax_i.set_yticklabels(tick_labels)
        plt.setp(ax_i.get_yticklabels(), bbox=bbox)

    # enumerating over each axis in fig2
    for i in range(len(ax)):
        ax[i].xaxis.set_major_locator(mtick.FixedLocator([i]))  # set tick locations along the x-axis
        set_ticks_for_axis(i, ax[i], n_ticks=10)  # set ticks along the y-axis

    # create a twin axis on the last subplot of fig2
    # this will enable you to label the last axis with y-ticks
    ax2 = plt.twinx(ax[-1])
    dim = len(ax)
    ax2.xaxis.set_major_locator(mtick.FixedLocator([x[-2], x[-1]]))
    set_ticks_for_axis(dim, ax2, n_ticks=10)
    ax2.set_xticklabels([objs[-2], objs[-1]])
    plt.setp(ax2.get_yticklabels(), bbox=bbox)

    # correction foe specific ax ticks
    values = [_ for _ in range(6, 25)]
    norm_values = [1 - ((24 - _) / (24 - 6)) for _ in values]
    ax[1].set_yticks(norm_values)
    ax[1].set_yticklabels(values)

    values = [_ for _ in range(0, 24)]
    norm_values = [1 - ((23 - _) / (23 - 0)) for _ in values]
    ax[2].set_yticks(norm_values)
    ax[2].set_yticklabels(values)

    plt.subplots_adjust(wspace=0, hspace=0.2, left=0.06, right=0.9, bottom=0.13, top=0.95)
    sm_good = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm_good.set_array([])
    cbar = plt.colorbar(sm_good, ax=ax, orientation='vertical', label="Coordinated Distributed LS Reduction (%)",
                        fraction=0.02, pad=0.07)
    min_val, max_val, v_range = min_max_range[objs[0]]
    locations = np.linspace(0, 1, 10)
    tick_labels = [round(_ * v_range + min_val, 2) for _ in locations]
    cbar.set_ticks(locations)
    cbar.set_ticklabels(tick_labels)


def area_plot(data):
    df = data.sort_values(by="decentralized", ascending=True)[['decentralized', 'coordinated_distributed', 'centralized']]
    fig, ax = plt.subplots()
    ax.bar(range(len(df)), df.iloc[:, :]['decentralized'], alpha=0.8, color=COLORS[2], width=1, label="Decentralized")
    ax.bar(range(len(df)), df.iloc[:, :]['coordinated_distributed'], alpha=0.8, color=COLORS[0], width=1, label="Coord-Distrib")
    ax.bar(range(len(df)), df.iloc[:, :]['centralized'], alpha=0.8, color=COLORS[1], width=1, label="Centralized")

    ax.grid()
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_ylabel("Total LS (kWhr)")

    df['Coord-Distrib'] = df['coordinated_distributed'] - df['decentralized']
    df['Centralized'] = df['centralized'] - df['decentralized']
    df = df[['Centralized', 'Coord-Distrib']]
    df = df[df['Coord-Distrib'] <= 0] * -1
    df = df.sort_values('Centralized')
    df.reset_index(inplace=True, drop=True)

    # ax = df.plot.area()
    # ax.set_xlabel('Scenario')
    # ax.set_ylabel("LS Reduction (kWhr)")


def compare_strategies(data):
    fig, ax = plt.subplots()
    sc = ax.scatter(data["central_diff"] * -1, data["coord_dist_diff"] * -1, alpha=1, edgecolor='k', linewidth=0.5,
                    c=data['decentralized'], cmap="RdYlBu_r")

    ax.grid()
    ax.set_xlabel("Centralized LS reduction (kWhr)")
    ax.set_ylabel("Coordinated Distributed LS reduction (kWhr)")
    ax.set_axisbelow(True)
    plt.colorbar(sc, label="Decentralized LS (kWhr)")


if __name__ == "__main__":
    pds = PDS("data/pds_emergency_futurized")
    wds = WDS("data/wds_wells")
    results_file = "20240605-104624_output.csv"
    data = load_results(results_file)
    # box(data)
    # ls_reduction(data, explanatory_var="decentralized", x_label="Load Shedding - Independent Decision Making (kWhr)")
    # ls_reduction(data, explanatory_var="pv_factor", x_label="PV Factor")
    # scatter_hist(data, explanatory_var="decentralized", x_label="Load Shedding - Independent Decision Making (kWhr)")
    # analyze_costs(data)
    # double_factor_scatter(data, 't1_state', 't', z_col='coord_dist_reduction')
    # all_factors(data)
    # kde_plot()
    # parallel_coords()
    # mpl_parallel_coordinates(data)
    # area_plot(data)
    # compare_strategies(data)

    # plt.plot(data[['decentralized', 'coordinated_distributed', 'centralized']].mean())

    plt.show()
