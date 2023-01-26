import haiku as hk
from jax import numpy as jnp, random
from ramsey.covariance_functions import ExponentiatedQuadratic, Linear
from ramsey.train import train_sparse_gaussian_process
from ramsey.models import SparseGP

import matplotlib.pyplot as plt

from case_study_data import rng, n_series, n, n_train, x_train, y_train, x_test, y_test
from case_study_utils import plot, point_forecast_eval
from case_study_config import *

def train_sgp(key, _sgp, y, x):
    init_key, train_key = random.split(key, 2)
    sgp = hk.transform(_sgp)
    params = sgp.init(init_key, x=x, y=y)

    params, _ = train_sparse_gaussian_process(
        sgp,
        params,
        train_key,
        x=x,
        y=y,
        stepsize=1e-3,
        n_iter=5000
    )

    return sgp, params

def predict_sgp(key, model, params, x_star, x_train, y_train):

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

m = 50

fig, axes = plt.subplots(figsize=(15, 10), nrows=2, ncols=2)

fig.suptitle(f'Sparse GP\nTraining Points: n={n_train}, Inducing Points: m={m}', fontsize=fontsize)

for _, (idx, ax) in enumerate(zip(range(n_series), axes.flatten())):
    _x_train = x_train[idx]
    _y_train = y_train[idx]
    _x_test = x_test[idx]
    _y_test = y_test[idx]

    _x_star = jnp.concatenate((_x_train, _x_test))

    def _sparse_gaussian_process(**kwargs):
        rho_init= hk.initializers.Constant(jnp.log(0.025))
        kernel = Linear() + ExponentiatedQuadratic(rho_init=rho_init)
        sgp = SparseGP(kernel, m)
        return sgp(**kwargs)

    sparse_gaussian_process, params = train_sgp(next(rng), _sparse_gaussian_process, _y_train, _x_train)

    mean, sigma = predict_sgp(next(rng), sparse_gaussian_process, params, _x_star, _x_train, _y_train)

    ax = plot(ax, mean, sigma,_y_train, _x_train, _y_test, _x_test)
    ax.set_title('Timeseries %d' % (_+1), fontsize=fontsize)

    _y =  jnp.concatenate((_y_train, _y_test))
    _y_star = jnp.reshape(mean, (n,1))

    _sMAPE, _MASE =  point_forecast_eval(_y_star, _y, n_train, 24)

    print('sMAPE: %.4f' % (_sMAPE*100))
    print('MASE:  %.4f' % (_MASE))

    sMAPE.append(_sMAPE)
    MASE.append(_MASE)

sMAPE = jnp.mean(jnp.asarray(sMAPE))
MASE = jnp.mean(jnp.asarray(MASE))

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