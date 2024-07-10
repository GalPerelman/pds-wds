import copy
import datetime
import os
import random

from gurobipy import gurobipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import graphs
import utils
from pds import PDS
from opt import Optimizer
from wds import WDS

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class Scenario:
    def __init__(self, n_tanks: int, n_batteries: int, power_lines: list, max_outage_lines: int, **kwargs):
        self.n_tanks = n_tanks
        self.n_batteries = n_batteries
        self.power_lines = power_lines
        self.max_outage_lines = max_outage_lines
        self.kwargs = kwargs

        # default parameters - standard scenario
        self.t = 24
        self.start_time = 0
        self.wds_demand_factor = 1
        self.pds_demand_factor = 1
        self.pv_factor = 1
        self.outage_lines = []
        self.tanks_state = np.ones(shape=(self.n_tanks,))
        self.batteries_state = np.ones(shape=(self.n_batteries,))

        # set specified values if passed:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def draw_random(self):
        """
        Overwrite defaults but maintain specified values that were passed to class initiation
        """
        outage_set = utils.get_subsets_of_max_size(elements=self.power_lines, max_subset_size=self.max_outage_lines)
        rand_params = {
            "t": np.random.randint(low=6, high=25),
            "start_time": np.random.randint(low=0, high=24),
            "wds_demand_factor": round(np.random.uniform(low=0.8, high=1.2), ndigits=4),
            "pds_demand_factor": round(np.random.uniform(low=0.8, high=1.2), ndigits=4),
            "pv_factor": round(np.random.uniform(low=0.8, high=1.2), ndigits=4),
            "outage_lines": outage_set[np.random.randint(low=0, high=max(1, len(outage_set)))],
            "tanks_state": np.random.uniform(low=0.2, high=1, size=self.n_tanks).round(4),
            "batteries_state": np.random.uniform(low=0.2, high=1, size=self.n_batteries).round(4),
        }

        for param, rand_value in rand_params.items():
            setattr(self, param, self.kwargs.get(param, rand_params[param]))


class CommunicateProtocolMaxInfo:
    """
    In this communication protocol, the power utility has access to WDS optimization models
    The power utility can solve a conjunctive problem and deliver pumps penalties based on it
    """

    def __init__(self, pds_data, wds_data, scenario):
        self.pds_data = pds_data
        self.wds_data = wds_data
        self.scenario = scenario

    def get_pumps_penalties(self, mip_gap):
        # run centralized coupling - assuming max information sharing, power utility can run centralized coupling model
        model = opt_resilience(pds_data=self.pds_data, wds_data=self.wds_data, scenario=self.scenario, display=False,
                               mip_gap=mip_gap)
        # get pumps schedule
        pumps_penalties = model.wds.pumps_combs @ model.x['pumps'].get()
        pumps_penalties = np.ones(pumps_penalties.shape) - pumps_penalties
        return pumps_penalties


class CommunicateProtocolBasic:
    """
        In this communication protocol, the power utility has only the standard pumps schedule
        Standard pumps schedule - was delivered as plan from water utility or estimated based on historic pumping loads
        The power utility solves an inner optimization problem to evaluate the pumps penalties
        """

    def __init__(self, pds_data, wds_data, scenario):
        self.pds_data = pds_data
        self.wds_data = wds_data
        self.scenario = scenario

    def get_wds_standard(self, mip_gap):
        """
        wds standard schedule - based on wds cost minimization
        """
        try:
            model_wds = Optimizer(self.pds_data, self.wds_data, scenario=self.scenario, display=False)
            model_wds.build_water_problem(obj="cost", final_tanks_ratio=1, w=1)
            model_wds.solve(mip_gap)
            return model_wds.x['pumps'].get()
        except RuntimeError:
            return

    def get_pumps_penalties(self, mip_gap):
        # get the standard wds operation
        x_pumps = self.get_wds_standard(mip_gap)
        if x_pumps is None:
            return

        else:
            # solve inner pds problem
            model = Optimizer(pds_data=self.pds_data, wds_data=self.wds_data, scenario=self.scenario, display=False)
            model.build_inner_pds_problem(x_pumps=x_pumps)
            for line_idx in self.scenario.outage_lines:
                model.disable_power_line(line_idx)
            model.solve(mip_gap)

            pumps_penalties = model.wds.pumps_combs @ model.x['pumps'].get()
            pumps_penalties = np.ones(pumps_penalties.shape) - pumps_penalties
            return pumps_penalties


class Simulation:
    def __init__(self, pds_data, wds_data, opt_display, final_tanks_ratio, comm_protocol, rand_scenario,
                 scenario_const=None):

        self.pds_data = pds_data
        self.wds_data = wds_data
        self.opt_display = opt_display
        self.final_tanks_ratio = final_tanks_ratio
        self.comm_protocol = comm_protocol
        self.rand_scenario = rand_scenario

        if scenario_const is None:
            self.scenario_const = {}
        else:
            self.scenario_const = scenario_const

        # initiate pds and wds objects for data usage in simulation functions
        self.base_pds = PDS(self.pds_data)
        self.base_wds = WDS(self.wds_data)
        self.scenario = self.draw_scenario()

        self.central_coupled_model = None
        self.decoupled_model = None
        self.coordinated_model = None

    def draw_scenario(self):
        s = Scenario(n_tanks=self.base_wds.n_tanks,
                     n_batteries=self.base_pds.n_batteries,
                     power_lines=self.base_pds.lines.index.to_list(),
                     max_outage_lines=2,
                     **self.scenario_const
                     )

        if self.rand_scenario:
            s.draw_random()
        else:
            pass
        return s

    def run_decoupled(self, wds_objective, mip_gap):
        # INDIVIDUAL OPERATION (Independent) - WDS IS NOT AWARE TO POWER EMERGENCY AND OPTIMIZES FOR 24 HR
        scenario = copy.deepcopy(self.scenario)
        scenario.t = 24
        model_wds = Optimizer(pds_data=self.pds_data, wds_data=self.wds_data, scenario=scenario,
                              display=self.opt_display)

        model_wds.build_water_problem(obj=wds_objective)
        model_wds.solve(mip_gap)
        if model_wds.status == gurobipy.GRB.INFEASIBLE or model_wds.status == gurobipy.GRB.INF_OR_UNBD:
            model = Optimizer(pds_data=self.pds_data, wds_data=self.wds_data, scenario=self.scenario,
                              display=self.opt_display)

            model.objective = None
            model.status = gurobipy.GRB.INFEASIBLE

        else:
            # planned schedule (can be seen also as historic nominal schedule)
            # solves for 24 hours but take only the scenario duration first steps for comparison purposes
            x_pumps = model_wds.x['pumps'].get()[:, :self.scenario.t]

            # Decoupled (no collaboration) - solve resilience problem with given WDS operation
            model = Optimizer(pds_data=self.pds_data, wds_data=self.wds_data, scenario=self.scenario,
                              display=self.opt_display)
            for line_idx in self.scenario.outage_lines:
                model.disable_power_line(line_idx)
            model.build_combined_resilience_problem(x_pumps=x_pumps)
            model.solve(mip_gap)

        return model

    def run_centralized_coupled(self, mip_gap):
        model = Optimizer(pds_data=self.pds_data, wds_data=self.wds_data, scenario=self.scenario,
                          display=self.opt_display)
        for line_idx in self.scenario.outage_lines:
            model.disable_power_line(line_idx)
        model.build_combined_resilience_problem()
        model.solve(mip_gap)
        return model

    def run_coordinated(self, comm_protocol, mip_gap):
        p = comm_protocol(self.pds_data, self.wds_data, self.scenario)
        w = p.get_pumps_penalties(mip_gap)
        if w is None:
            model = opt_resilience(self.pds_data, self.wds_data, self.scenario, self.opt_display, mip_gap=mip_gap)
            model.objective = None
            model.status = gurobipy.GRB.INFEASIBLE

        else:
            model_wds = Optimizer(self.pds_data, self.wds_data, scenario=self.scenario, display=False)
            model_wds.build_water_problem(obj="emergency", final_tanks_ratio=self.final_tanks_ratio, w=w)
            model_wds.solve(mip_gap)
            x_pumps = model_wds.x['pumps'].get()  # planned schedule (can be seen also as historic nominal schedule)
            model = opt_resilience(self.pds_data, self.wds_data, self.scenario, self.opt_display, mip_gap,
                                   x_pumps=x_pumps)

        return model

    def run_and_record(self, mip_gap):
        self.central_coupled_model = self.run_centralized_coupled(mip_gap=mip_gap)
        self.decoupled_model = self.run_decoupled(wds_objective="cost", mip_gap=mip_gap)
        self.coordinated_model = self.run_coordinated(self.comm_protocol, mip_gap=mip_gap)

        try:
            decoupled_wds_cost = self.decoupled_model.get_systemwise_costs(self.scenario.t)[0]
            coordinated_wds_cost = self.coordinated_model.get_systemwise_costs(self.scenario.t)[0]
            decoupled_wds_penalties = list(self.decoupled_model.x['penalty_final_vol'].get().T[0])
            coordinated_wds_penalties = list(self.coordinated_model.x['penalty_final_vol'].get().T[0])
            decoupled_wds_pumped_vol = (
                self.decoupled_model.wds.get_pumped_vol(self.decoupled_model.x['pumps'].get())).sum()
            coordinated_wds_pumped_vol = (
                self.coordinated_model.wds.get_pumped_vol(self.coordinated_model.x['pumps'].get())).sum()
            decoupled_final_vol = self.decoupled_model.x['vol'].get()[:, self.scenario.t - 1]
            coordinated_final_vol = self.coordinated_model.x['vol'].get()[:, -1]

            central_coupled_ls_ts = (self.central_coupled_model.x['penalty_p'].get() * self.base_pds.pu_to_kw).flatten()
            coordinated_ls_ts = (self.coordinated_model.x['penalty_p'].get() * self.base_pds.pu_to_kw).flatten()
            decoupled_ls_ts = (self.decoupled_model.x['penalty_p'].get() * self.base_pds.pu_to_kw).flatten()

        except (RuntimeError, AttributeError):
            decoupled_wds_cost = None
            coordinated_wds_cost = None
            decoupled_wds_penalties = None
            coordinated_wds_penalties = None
            decoupled_wds_pumped_vol = None
            coordinated_wds_pumped_vol = None
            decoupled_final_vol = None
            coordinated_final_vol = None
            central_coupled_ls_ts = None
            coordinated_ls_ts = None
            decoupled_ls_ts = None

        return (
            {
                "centralized_coupled": self.central_coupled_model.objective,
                "decoupled": self.decoupled_model.objective,
                "coordinated": self.coordinated_model.objective,
                "decoupled_wds_penalties": decoupled_wds_penalties,
                "coordinated_wds_penalties": coordinated_wds_penalties,
                "decoupled_wds_cost": decoupled_wds_cost,
                "coordinated_wds_cost": coordinated_wds_cost,
                "decoupled_wds_vol": decoupled_wds_pumped_vol,
                "coordinated_wds_vol": coordinated_wds_pumped_vol,
                "decoupled_final_vol": decoupled_final_vol,
                "coordinated_final_vol": coordinated_final_vol,
                "t": self.scenario.t,
                "start_time": self.scenario.start_time,
                "wds_demand_factor": self.scenario.wds_demand_factor,
                "pds_demand_factor": self.scenario.pds_demand_factor,
                "pv_factor": self.scenario.pv_factor,
                "outage_lines": self.scenario.outage_lines,
                "n_outage_lines": len(self.scenario.outage_lines),
                "tanks_state": self.scenario.tanks_state,
                "batteries_state": self.scenario.batteries_state,
                "final_tanks_ratio": self.final_tanks_ratio
            },
            {"cantral_ls": central_coupled_ls_ts, "coord_dist_ls": coordinated_ls_ts, "decantral_ls": decoupled_ls_ts}
        )

    def plot_wds(self):
        pumps_names = [col for col in self.base_wds.combs.columns if col.startswith("pump_")]
        fig_gantt, axes = plt.subplots(nrows=2, sharex=True)

        g = graphs.OptGraphs(self.decoupled_model)
        ax_decoupled = g.pumps_gantt(pumps_names=pumps_names, title='', ax=axes[0])
        ax_decoupled.set_title("Decoupled")
        fig = g.plot_all_tanks(leg_label="Decoupled")
        fig_bat = g.plot_batteries(leg_label="Decoupled")
        fig_gen = g.plot_all_generators(leg_label="Decoupled")
        fig_power = g.pump_results(pumps_names=pumps_names, leg_label="Decoupled")

        g = graphs.OptGraphs(self.coordinated_model)
        ax_coordinated = g.pumps_gantt(pumps_names=pumps_names, title='', ax=axes[1])
        ax_coordinated.set_title("Coordinated")
        fig = g.plot_all_tanks(fig=fig, leg_label="Coordinated")
        fig_bat = g.plot_batteries(leg_label="Coordinated", fig=fig_bat)
        fig_gen = g.plot_all_generators(leg_label="Coordinated", fig=fig_gen)
        fig_power = g.pump_results(pumps_names=pumps_names, leg_label="Coordinated", fig=fig_power, axes_legend=True)

        fig_gantt.subplots_adjust(left=0.13, bottom=0.15, right=0.92, top=0.9, hspace=0.35)
        fig_gantt.text(0.5, 0.04, 'Time (hr)', ha='center')

        fig_power.text(0.5, 0.04, 'Time (hr)', ha='center')
        fig_power.text(0.04, 0.5, 'Power (kW)', va='center', rotation='vertical')

        # for unified figure legend:
        # handles, labels = fig_power.axes[-1].get_legend_handles_labels()
        # fig_power.legend(handles, labels, loc='center left', bbox_to_anchor=(0.05, 0.05), ncol=2)
        fig_power.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.9, hspace=0.3)


def opt_resilience(pds_data, wds_data, scenario, display, mip_gap, x_pumps=None):
    model = Optimizer(pds_data=pds_data, wds_data=wds_data, scenario=scenario, display=display)
    for line_idx in scenario.outage_lines:
        model.disable_power_line(line_idx)
    model.build_combined_resilience_problem(x_pumps=x_pumps)
    model.solve(mip_gap)
    return model


def analyze_single_scenario(pds_data, wds_data, results_df: pd.DataFrame, idx: int, mip_gap, opt_display):
    scenario = results_df.iloc[idx]

    scenario_params = {
        "t": scenario["t"],
        "start_time": scenario["start_time"],
        "wds_demand_factor": scenario["wds_demand_factor"],
        "pds_demand_factor": scenario["pds_demand_factor"],
        "pv_factor": scenario["pv_factor"],
        "outage_lines": scenario["outage_lines"],
        "tanks_state": np.array(scenario["tanks_state"]),
        "batteries_state": np.array(scenario["batteries_state"])
    }

    sim = Simulation(pds_data=pds_data, wds_data=wds_data, opt_display=opt_display, final_tanks_ratio=0.2,
                     comm_protocol=CommunicateProtocolBasic, rand_scenario=False, scenario_const=scenario_params)
    sim_results, time_series_ls = sim.run_and_record(mip_gap=mip_gap)
    sim.plot_wds()


def run_random_scenarios(pds_data, wds_data, n, final_tanks_ratio, mip_gap, export_path=''):
    results = []
    central_coupled_ls = pd.DataFrame()
    coordinated = pd.DataFrame()
    decoupled_ls = pd.DataFrame()
    for _ in range(n):

        sim = Simulation(pds_data=pds_data, wds_data=wds_data, opt_display=False,
                         final_tanks_ratio=final_tanks_ratio, comm_protocol=CommunicateProtocolBasic,
                         rand_scenario=True)

        sim_results, time_series_ls = sim.run_and_record(mip_gap=mip_gap)
        sim_results = utils.convert_arrays_to_lists(sim_results)
        results.append(sim_results)

        central_coupled_ls = pd.concat([central_coupled_ls, pd.DataFrame(time_series_ls['cantral_ls']).T], axis=0)
        coordinated = pd.concat([coordinated, pd.DataFrame(time_series_ls['coord_dist_ls']).T], axis=0)
        decoupled_ls = pd.concat([decoupled_ls, pd.DataFrame(time_series_ls['decantral_ls']).T], axis=0)

        if export_path:
            idx = len(results)
            export_df(pd.DataFrame([sim_results], index=[idx]), export_path)
            export_df(pd.DataFrame([time_series_ls['cantral_ls']], index=[idx]).T, export_path[:-4] + "_cantral_ls.csv")
            export_df(pd.DataFrame([time_series_ls['coord_dist_ls']], index=[idx]).T, export_path[:-4] + "_coord_dist_ls.csv")
            export_df(pd.DataFrame([time_series_ls['decantral_ls']], index=[idx]).T, export_path[:-4] + "_decantral_ls.csv")


def export_df(df, path):
    # with open(path, 'w', newline='') as file:
    #     df.to_csv(file)
    df.to_csv(path, mode='a', header=not os.path.exists(path))