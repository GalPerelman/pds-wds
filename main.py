import os
import code
import matplotlib.pyplot as plt

from optimization import Opt


if __name__ == "__main__":
    PDS_DATA = os.path.join('data', 'pds')
    WDS_DATA = os.path.join('data', 'wds')
    opt = Opt(pds_data=PDS_DATA, wds_data=WDS_DATA, T=24)
    opt.solve()
    opt.plot_results(t=0)

    # a = utils.get_mat_for_type(opt.wds.pipes, 'pump')
    # dh = a @ opt.x['f'] * opt.wds.pipes['R'].values.reshape(-1, 1)
    # print(dh)
    plt.show()
    # code.interact(local=locals())
