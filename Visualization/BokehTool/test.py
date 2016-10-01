import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from GPEclass import *
import seaborn as sns
import pandas as pd
import numpy as np


def init_speed_test():
    """
    Tests Speed of initialization
    Produces 3 plots
    :return:
    """

    n_features = []
    n_samples = []
    data = []
    color = []
    controls = []
    rep = []

    for r in range(5):
        for n_plot in [1,2,3]:
            for n_sample in [1e2,1e3,1e4,1e5]:

                n_sample = int(n_sample)

                n_samples.append(n_sample)
                n_features.append(n_plot*2)


                gpe = GPE(offline=True,
                          test_n_plots=n_plot,
                          test_n_samples=n_sample)

                data.append(np.log(gpe.init_data_time))
                color.append(np.log(gpe.init_color_time))
                controls.append(np.log(gpe.init_control_time))
                rep.append(r)

    time_df = pd.DataFrame({'n_samples':n_samples,
                            'n_features':n_features,
                            'data':data,
                            'color':color,
                            'controls':controls,
                            'rep':rep})

    print(time_df)
    g1 = sns.factorplot(x="n_samples", y="color", hue="n_features", data=time_df)
    plt.savefig('Init color Speed Test 1.jpg')

    g2 = sns.factorplot(x="n_features", y="color", hue="n_samples", data=time_df)
    plt.savefig('Init color Speed Test 2.jpg')
def update_speed_test():
    """
    Tests Speed of Selection and Color updates
    Produces 3 plots
    :return:
    """
    n_features = []
    n_samples = []
    n_selected = []
    update = []
    rep = []

    for n_select in [0,25,50]:
        for n_plot in [1, 2, 3]:
            for n_sample in [1e2, 1e3, 1e4, 1e5]:

                n_sample = int(n_sample)
                gpe = GPE(offline=True,
                          test_n_plots=n_plot,
                          test_n_samples=n_sample)

                for r in range(5):

                    n_selected.append(n_select)

                    n_samples.append(n_sample)
                    n_features.append(n_plot * 2)
                    gpe.pseudo_update(n_select)
                    update.append(np.log(gpe.update_time))

                    rep.append(r)

    time_df = pd.DataFrame({'n_samples': n_samples,
                            'n_features': n_features,
                            'update': update,
                            'n_selected': n_selected,
                            'rep': rep})

    print(time_df)
    g1 = sns.factorplot(x="n_samples", y="update", hue="n_features",col='n_selected', data=time_df)
    plt.savefig('Update Speed Test 1.jpg')

    g2 = sns.factorplot(x="n_features", y="update", hue="n_samples",col='n_selected', data=time_df)
    plt.savefig('Update Speed Test 2.jpg')

    g3 = sns.factorplot(x="n_selected", y="update", hue="n_samples", col='n_features', data=time_df)
    plt.savefig('Update Speed Test 3.jpg')




if __name__ == '__main__':
    update_speed_test()
    #init_speed_test()
