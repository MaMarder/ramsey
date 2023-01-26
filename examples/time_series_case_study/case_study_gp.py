import haiku as hk
from jax import numpy as jnp, random
from ramsey.covariance_functions import ExponentiatedQuadratic, Periodic
from ramsey.train import train_gaussian_process
from ramsey.models import GP

import matplotlib.pyplot as plt

from case_study_data import rng, n_series, n, n_train, x_train, y_train, x_test, y_test
from case_study_utils import plot, point_forecast_eval
from case_study_config import *

def train_gp(key, _gp, y, x):
    gp = hk.transform(_gp)
    params = gp.init(key, x=x)

    params, _ = train_gaussian_process(
        gp,
        params,
        key,
        x=x,
        y=y,
        stepsize=1e-3,
        n_iter=5000, 
        verbose=True
    )

    return gp, params

def predict_gp(key, model, params, x_star, x_train, y_train):

    posterior_dist = model.apply(
        params=params,
        rng=key,
        x=x_train,
        y=y_train,
        x_star=x_star,
    )
    mean = posterior_dist.mean()
    sigma = posterior_dist.stddev()

    return mean, sigma


sMAPE = []
MASE = []

series_cycle = [24,24,24,10]

fig, axes = plt.subplots(figsize=(15, 10), nrows=2, ncols=2)

for _, (idx, ax) in enumerate(zip(range(n_series), axes.flatten())):
    _x_train = x_train[idx]
    _y_train = y_train[idx]
    _x_test = x_test[idx]
    _y_test = y_test[idx]

    _x_star = jnp.concatenate((_x_train, _x_test))

    p = series_cycle[_] * _x_train[1,1]

    def _gaussian_process(**kwargs):
        
        kernel = Periodic(p) * ExponentiatedQuadratic() + Periodic(p) + ExponentiatedQuadratic()
        gp = GP(kernel)
        return gp(**kwargs)


    gaussian_process, params = train_gp(next(rng), _gaussian_process, _y_train, _x_train)

    mean, sigma = predict_gp(next(rng), gaussian_process, params, _x_star, _x_train, _y_train)

    ax = plot(ax, mean, sigma,_y_train, _x_train, _y_test, _x_test)
    ax.set_title('Timeseries %d' % (_+1), fontsize=fontsize)

    _y =  jnp.concatenate((_y_train, _y_test))
    _y_star = jnp.reshape(mean, (n,1))

    _sMAPE, _MASE =  point_forecast_eval(_y_star, _y, n_train, 24)

    print('sMAPE: %.4f' % (_sMAPE*100))
    print('MASE:  %.4f' % (_MASE))

    sMAPE.append(_sMAPE)
    MASE.append(_MASE)

sMAPE = jnp.mean(jnp.squeeze(jnp.stack([_ for _ in sMAPE])), axis = 0)
MASE = jnp.mean(jnp.squeeze(jnp.stack([_ for _ in MASE])), axis = 0)

print('sMAPE: %.4f' % (sMAPE*100))
print('MASE:  %.4f' % (MASE))

handles, labels = ax.get_legend_handles_labels()
plt.legend( handles, labels,
            bbox_to_anchor=(0.5, -0.005),
            loc="lower center",
            bbox_transform=fig.transFigure, 
            ncol=4, 
            frameon=True, fontsize=fontsize, facecolor='white', framealpha=1)
plt.show()