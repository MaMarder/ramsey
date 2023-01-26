import haiku as hk
from jax import numpy as jnp, random
from ramsey.train import train_neural_process
from ramsey.models import NP

import matplotlib.pyplot as plt

from case_study_data import rng, n_series, n_train, n, x_train, y_train, x_test, y_test
from case_study_utils import plot, point_forecast_eval
from case_study_config import *

def _neural_process(**kwargs):
    dim = 128
    np = NP(
        decoder=hk.nets.MLP([dim]*3+ [2]),
        latent_encoder=(hk.nets.MLP([dim]*3), hk.nets.MLP([dim, dim * 2]))
    )
    return np(**kwargs)

def train_np(rng, x, y, n_context, n_target):
    neural_process = hk.transform(_neural_process)
    params = neural_process.init(
        next(rng), x_context=x, y_context=y, x_target=x
    )
    params, _ = train_neural_process(
        neural_process,
        params,
        next(rng),
        x=x,
        y=y,
        n_context=n_context,
        n_target=n_target,
        n_iter=200000,
        verbose=True
    )

    return neural_process, params


def predict_np(key, model, params, x_star, x_context, y_context):

    pred_dist = model.apply(
        params=params,
        rng=key, 
        x_context=x_context[jnp.newaxis, :], 
        y_context=y_context[jnp.newaxis, :],
        x_target=x_star[jnp.newaxis, :]) 

    mean = pred_dist.mean
    sigma = pred_dist.scale

    return jnp.squeeze(mean), jnp.squeeze(sigma)

n_context = int((n_train - 1) * 0.5) 
n_target = int((n_train - 1) * 0.5)

neural_process, params = train_np(
    rng, 
    x_train, 
    y_train, 
    n_context,
    n_target
)


sample_idxs = random.choice(
    next(rng),
    x_train.shape[1],
    shape=(n_context + n_target,),
    replace=False,
)

sMAPE = []
MASE = []


fig, axes = plt.subplots(figsize=(15, 10), nrows=2, ncols=2)
for _, (idx, ax) in enumerate(zip(range(n_series), axes.flatten())):

    _x_train = x_train[idx]
    _y_train = y_train[idx]
    _x_test = x_test[idx]
    _y_test = y_test[idx]

    _x_star = jnp.concatenate((_x_train, _x_test))

    _x_context = _x_train[sample_idxs]
    _y_context = _y_train[sample_idxs]


    mean, sigma = predict_np(next(rng), neural_process, params, _x_star, _x_context, _y_context)

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
