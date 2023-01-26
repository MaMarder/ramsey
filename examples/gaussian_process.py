"""
Gaussian process regression
===========================

This example implements the training and prediction of a Gaussian process
regression model.

References
----------

[1] Rasmussen, Carl E and Williams, Chris KI.
    "Gaussian Processes for Machine Learning". MIT press, 2006.
"""

import haiku as hk

from jax import numpy as jnp, random
import matplotlib.pyplot as plt

from ramsey.train import train_gaussian_process
from ramsey.data import sample_from_gaussian_process
from ramsey.covariance_functions import ExponentiatedQuadratic, Linear
from ramsey.models import GP

from jax.config import config
config.update("jax_enable_x64", True)

def data(key, rho, sigma, n=1000):
    (x_target, y_target), f_target = sample_from_gaussian_process(
        key, batch_size=1, num_observations=n, rho=rho, sigma=sigma
    )
    return (x_target.reshape(n, 1), y_target.reshape(n, 1)), f_target.reshape(
        n, 1
    )

def _gaussian_process(**kwargs):
    kernel = Linear() + ExponentiatedQuadratic()
    gp = GP(kernel)
    return gp(**kwargs)

def train_gp(key, x, y):
    _, init_key, train_key = random.split(key, 3)
    gaussian_process = hk.transform(_gaussian_process)
    params = gaussian_process.init(init_key, x=x)

    params, _ = train_gaussian_process(
        gaussian_process,
        params,
        train_key,
        x=x,
        y=y,
        verbose = True
    )

    return gaussian_process, params


def plot(key, gaussian_process, params, x, y, f, train_idxs):
    fig, ax = plt.subplots(figsize=(24, 9))
    srt_idxs = jnp.argsort(jnp.squeeze(x))
    ax.plot(
        jnp.squeeze(x)[srt_idxs],
        jnp.squeeze(f)[srt_idxs],
        color="black",
        alpha=0.5,
        label="Latent function " + r"$f \sim GP$"
    )
    ax.scatter(
        jnp.squeeze(x[train_idxs, :]),
        jnp.squeeze(y[train_idxs, :]),
        color="red",
        marker="+",
        s=50,
        alpha=0.75,
        label="Training data"
    )

    key, apply_key = random.split(key, 2)
    posterior_dist = gaussian_process.apply(
        params=params,
        rng=apply_key,
        x=x[train_idxs, :],
        y=y[train_idxs, :],
        x_star=x,
    )

    y_star = posterior_dist.mean()
    ax.plot(
        jnp.squeeze(x)[srt_idxs], 
        jnp.squeeze(y_star)[srt_idxs],
        color="blue", 
        alpha=0.75,
        label="Posterior mean"
    )

    sigma = posterior_dist.stddev()
    ucb = y_star + 1.644854 * sigma
    lcb = y_star - 1.644854 * sigma
    ax.fill_between(
        jnp.squeeze(x)[srt_idxs],
        lcb[srt_idxs], ucb[srt_idxs],
        color="grey", alpha=0.2,
        label=r'90% posterior interval'
    )

    ax.grid()
    ax.tick_params('both', labelsize=20)
    ax.set_frame_on(False)
    plt.legend( bbox_to_anchor=(0.5, -0.005),
                loc="lower center",
                bbox_transform=fig.transFigure, 
                ncol=4, 
                frameon=True, fontsize=20, facecolor='white', framealpha=1)
    plt.show()


def run():
    rng_seq = hk.PRNGSequence(123)
    n_train = 30

    (x, y), f = data(next(rng_seq), 0.25, 3.0)
    train_idxs = random.choice(
        next(rng_seq), jnp.arange(x.shape[0]), shape=(n_train,), replace=False
    )

    x_train, y_train = x[train_idxs, :], y[train_idxs, :]
    gaussian_process, params = train_gp(next(rng_seq), x_train, y_train)

    plot(
        next(rng_seq),
        gaussian_process,
        params,
        x,
        y,
        f,
        train_idxs,
    )


if __name__ == "__main__":
    run()
