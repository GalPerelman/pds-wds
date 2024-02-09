import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lp
import utils
from pds import PDS
import graphs
from lp import Optimizer, WDS

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
            "wds_demand_factor": np.random.uniform(low=0.8, high=1.2),
            "pds_demand_factor": np.random.uniform(low=0.75, high=1.25),
            "pv_factor": np.random.uniform(low=0.8, high=1.2),
            "outage_lines": outage_set[np.random.randint(low=0, high=len(outage_set))],
            "tanks_state": np.random.uniform(low=0.2, high=1, size=self.n_tanks),
            "batteries_state": np.random.uniform(low=0.2, high=1, size=self.n_batteries),
        }

        for param, rand_value in rand_params.items():
            setattr(self, param, self.kwargs.get(param, rand_params[param]))


class PumpsLogicPostpone:
    def __init__(self):
        pass


class PumpLogicConnectivity:
    def __init__(self, penalties_mat, pumps_bus_mat):
        self.penalties_mat = penalties_mat
        self.pumps_bus_mat = pumps_bus_mat

    def logic(self):
        pass


class Simulation:
    def __init__(self, pds_data, wds_data, t, opt_display):
        self.pds_data = pds_data
        self.wds_data = wds_data
        self.t = t
        self.opt_display = opt_display

        # initiate pds and wds objects for data usage in simulation functions
        self.pds = PDS(self.pds_data)
        self.wds = WDS(self.wds_data)

    def draw_random_emergency(self, n=1):
        pds = PDS(self.pds_data)
        outage_edges = pds.lines.sample(n=n)
        return outage_edges

    def run_individual(self, outage_lines, wds_objective, w=1):
        model_wds = Optimizer(pds_data=self.pds_data, wds_data=self.wds_data, t=self.t, display=self.opt_display)
        model_wds.build_water_problem(obj=wds_objective, w=w)
        model_wds.solve()
        x_pumps = model_wds.x['pumps'].get()

        # Non-cooperative - solve resilience problem with given WDS operation
        model = Optimizer(pds_data=self.pds_data, wds_data=self.wds_data, t=self.t, display=self.opt_display)
        for line_idx in outage_lines:
            model.disable_power_line(line_idx)
        model.build_combined_resilience_problem(x_pumps=x_pumps)
        model.solve()
        return model

    def run_cooperated(self, outage_lines):
        model = Optimizer(pds_data=self.pds_data, wds_data=self.wds_data, t=self.t, display=self.opt_display)
        for line_idx in outage_lines:
            model.disable_power_line(line_idx)
        model.build_combined_resilience_problem()
        model.solve()
        return model

    def get_penalties_for_pumps(self, model):
        """
        Get penalties for pumps based on load shedding results
        """
        # penalties according to bus shedding
        # pumps_penalties = (model.x['penalty_p'].get() * model.pds.pu_to_kw) * model.pds.bus_criticality.values

        # penalties according to desired pumps schedule
        # pumps_penalties is of (n_pumps x T) shape
        pumps_penalties = self.wds.pumps_combs @ model.x['pumps'].get()
        return pumps_penalties

    def get_lines_subsets(self, subset_size):
        pds = PDS(self.pds_data)
        return utils.get_subsets(pds.lines.index.to_list(), subset_size)

    def run_scenarios(self, subsets):
        """
        run a set of scenarios where subst is the group of disabled elements

        param:  subset_size - how many disabled lines
        param:  draw_method - all elements of subset_size if "all" or one random subset of size subset_size if "random"
        """
        df = pd.DataFrame()
        for subset in subsets:
            # run a fill cooperative problem - for reference
            joint_model = self.run_cooperated(outage_lines=subset)

            # run realistic situation where systems not communicate at all
            independent_model = self.run_individual(outage_lines=subset, wds_objective="cost")

            # get pds information to pass to wds
            if lp.GRB_STATUS[independent_model.status] != "OPTIMAL":
                # push pumping to future
                pumps_penalties = np.tile(np.arange(1, 0, -1 / 24), (independent_model.pds.n_bus, 1))
            else:
                pumps_penalties = self.get_penalties_for_pumps(joint_model)

            # wds based on information from pds
            communicate_model = self.run_individual(outage_lines=subset, wds_objective="emergency", w=pumps_penalties)

            # record results
            temp = pd.DataFrame(
                {"outage_set": subset,
                 "cooperative": f"{joint_model.objective} ({lp.GRB_STATUS[joint_model.status]})",
                 "independent": f"{independent_model.objective} ({lp.GRB_STATUS[independent_model.status]})",
                 "communicate": f"{communicate_model.objective} ({lp.GRB_STATUS[communicate_model.status]})"
                 }, index=[len(df)]).fillna('INF')

            df = pd.concat([df, temp])
            df.to_csv("one_line_outage.csv")
        print(df)

    def run(self, outage_lines):
        # Full cooperation
        model = Optimizer(pds_data=self.pds_data, wds_data=self.wds_data, t=self.t)
        for line_idx in outage_lines:
            model.disable_power_line(line_idx)
        model.build_combined_resilience_problem()
        model.solve()
        g = graphs.OptGraphs(model)
        g.plot_penalty()

        ########################################################################################################
        # solve WDS min cost problem - get the WDS policy when ignoring PDS
        model_wds = Optimizer(pds_data=self.pds_data, wds_data=self.wds_data, t=self.t)
        model_wds.build_water_problem(obj="cost")
        model_wds.solve()
        x_pumps = model_wds.x['pumps'].get()

        # Non-cooperative - solve resilience problem with given WDS operation
        model = Optimizer(pds_data=self.pds_data, wds_data=self.wds_data, t=self.t)
        for line_idx in outage_lines:
            model.disable_power_line(line_idx)(line_idx)
        model.build_combined_resilience_problem(x_pumps=x_pumps)
        model.solve()
        penalties_for_wds = utils.normalize_mat((model.x['penalty_p'].get())) * model.pds.bus_criticality.values

        ########################################################################################################
        # solve WDS emergency problem - trying to improve based on PDS information
        model_wds = Optimizer(pds_data=self.pds_data, wds_data=self.wds_data, t=self.t)
        model_wds.build_water_problem(obj="emergency", w=penalties_for_wds)
        model_wds.solve()
        x_pumps = model_wds.x['pumps'].get()

        # Partial-cooperative - solve resilience problem with given WDS operation
        model = Optimizer(pds_data=self.pds_data, wds_data=self.wds_data, t=self.t)
        for line_idx in outage_lines:
            model.disable_power_line(line_idx)
        model.build_combined_resilience_problem(x_pumps=x_pumps)
        model.solve()
        g = graphs.OptGraphs(model)
        g.plot_penalty()


if __name__ == "__main__":
    sim = Simulation(pds_data="data/pds_emergency_futurized", wds_data="data/wds_wells", t=24, opt_display=False)
    pds = PDS(sim.pds_data)
    one_line_outage = utils.get_subsets(pds.lines.index.to_list(), subsets_size=1)

    sim.run_scenarios(subsets=one_line_outage)
    plt.show()
