from copy import deepcopy

import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import matplotlib as mpl
import dvu

from imodels.spatial_shrinkage import SSTreeRegressor


dvu.set_style()

mpl.rcParams["figure.dpi"] = 250
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

cb2 = "#66ccff"
cb = "#1f77b4"
cr = "#cc0000"
cp = "#cc3399"
cy = "#d8b365"
cg = "#5ab4ac"


def plot_sim(n=50, std=1, reg_param=0, linear_data=False, return_curves=False):
    np.random.seed(13)

    if linear_data:

        def gt_func(X):
            return X

    else:

        def gt_func(X):
            return (
                +1 * (X < 2) * (X < 1)
                + -0 * (X < 2) * (X > 1)
                + +1 * (X >= 2) * (X < 3)
                + +0 * (X >= 2) * (X > 3)
            )

    # data to fit
    X = np.random.uniform(0, 4, n)
    X = np.sort(X)
    y = gt_func(X) + np.random.normal(0, 1, n) * std

    # data to plot
    X_tile = np.linspace(0, 4, 400)
    y_tile = gt_func(X_tile)

    m1 = DecisionTreeRegressor(random_state=1)  # , max_leaf_nodes=15)
    m1.fit(X.reshape(-1, 1), y)
    y_pred_dt = m1.predict(X_tile.reshape(-1, 1))

    mshrunk = SSTreeRegressor(deepcopy(m1), reg_param=reg_param)
    y_pred_shrunk = mshrunk.predict(X_tile.reshape(-1, 1))

    plt.plot(X, y, "o", color="black", ms=4, alpha=0.5, markeredgewidth=0)
    plt.plot(X_tile, y_tile, label="Groundtruth", color="black", lw=3)
    plt.plot(X_tile, y_pred_dt, "-", label="CART", color=cb, alpha=0.5, lw=4)
    plt.plot(X_tile, y_pred_shrunk, label="hsCART", color="#ff4b33", alpha=0.5, lw=4)
    plt.xlabel("X")
    plt.ylabel("Y")
    dvu.line_legend(adjust_text_labels=False)

    if return_curves:
        return X, y, X_tile, y_tile, y_pred_dt, y_pred_shrunk


plot_sim(n=400, std=1, reg_param=100)
plt.savefig("figs/intro_indicators.pdf")
plt.close()


plot_sim(n=400, std=1, reg_param=50, linear_data=True)
plt.savefig("figs/intro_linear.pdf")
plt.close()
