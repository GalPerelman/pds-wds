import os
import code
import matplotlib.pyplot as plt

from optimization import Opt


if __name__ == "__main__":
    PDS_DATA = os.path.join('data', 'pds')
    WDS_DATA = os.path.join('data', 'wds')
    opt = Opt(pds_data=PDS_DATA, wds_data=WDS_DATA, t=24, n=8)
    opt.solve()
    opt.plot_results(t=0)

    # code.interact(local=locals())
    plt.show()