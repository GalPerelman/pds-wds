import datetime
import os
import random

import numpy as np
import matplotlib.pyplot as plt

import emergency
import results_analysis

np.set_printoptions(suppress=True)


if __name__ == "__main__":
    global_seed = 42
    os.environ['PYTHONHASHSEED'] = str(global_seed)
    random.seed(global_seed)
    np.random.seed(global_seed)
    np.set_printoptions(suppress=True, precision=4)

    pds_data = "data/pds_emergency_futurized"
    wds_data = "data/wds_wells"

    # Main procedure - run 1000 extreme scenarios and compare the three strategies
    export_file = f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_output_bug_fix.csv'
    emergency.run_random_scenarios(pds_data=pds_data, wds_data=wds_data, n=1000, final_tanks_ratio=0.2,
                                   mip_gap=0.01, export_path=export_file)

    # Isolated factors analysis
    emergency.isolate_single_factor(pds_data=pds_data, wds_data=wds_data, factor='pv_factor', n=300,
                                    mip_gap=0.01, export_path="_pv_factor.csv")
    emergency.isolate_single_factor(pds_data=pds_data, wds_data=wds_data, factor='pds_demand_factor', n=300,
                                    mip_gap=0.01, export_path="_pds_demand_factor.csv")
    emergency.isolate_single_factor(pds_data=pds_data, wds_data=wds_data, factor='wds_demand_factor', n=300,
                                    mip_gap=0.01, export_path="_wds_demand_factor.csv")
    emergency.isolate_single_factor(pds_data=pds_data, wds_data=wds_data, factor='tanks_state', n=300,
                                    mip_gap=0.01, export_path="_tanks_state.csv")
    emergency.isolate_single_factor(pds_data=pds_data, wds_data=wds_data, factor='batteries_state', n=300,
                                    mip_gap=0.01, export_path="_batteries_state.csv")

    # Analyze specific scenarios
    data = results_analysis.load_results("20240605-104624_output.csv", drop_nans=False)
    emergency.analyze_single_scenario(pds_data, wds_data, data, 673, mip_gap=0.0001, opt_display=True)
    emergency.analyze_single_scenario(pds_data, wds_data, data, 151, mip_gap=0.0001, opt_display=True)

    plt.show()



