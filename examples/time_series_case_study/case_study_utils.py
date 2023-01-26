from jax import numpy as jnp
from case_study_config import *

def plot(ax, mean, sigma, y_train, x_train, y_test, x_test):

    x_star = jnp.concatenate((x_train, x_test))

    ucb = mean + 1.644854 * sigma
    lcb = mean - 1.644854 * sigma
    
    ax.plot(jnp.squeeze(x_train), jnp.squeeze(y_train),color="black", alpha=1)
    ax.plot(
        jnp.squeeze(x_test), jnp.squeeze(y_test), color="black", alpha=1, linestyle="dashed", label='Test data')
    ax.axvline(x_test[0], color='red')
    ax.scatter(
        jnp.squeeze(x_train), jnp.squeeze(y_train),
        color="red", marker="+", alpha=a_train, label='Train data')
    ax.plot(
        jnp.squeeze(x_star), jnp.squeeze(mean), 
        color="blue", alpha=a_pred_mean,label='Posterior mean')
    ax.fill_between(
        jnp.squeeze(x_star), lcb, ucb, 
        color="grey", alpha=a_post_int, label='90% Posterior interval')
    ax.grid()
    ax.set_frame_on(False)
    ax.tick_params('both', labelsize=fontsize)
    return ax

def smape(y_star, y):
    """
    Calculates sMAPE
    :param a: actual values
    :param b: predicted values
    :return: sMAPE

    Reference:
    https://github.com/Mcompetitions/M4-methods/blob/master/ML_benchmarks.py
    """
    y = jnp.reshape(y, (-1,))
    y_star = jnp.reshape(y_star, (-1,))
    return jnp.mean(2.0 * jnp.abs(y - y_star) / (jnp.abs(y) + jnp.abs(y_star))).item()

def mase(insample, y_test, y_hat_test, freq):
    """
    Calculates MAsE
    :param insample: insample data
    :param y_test: out of sample target values
    :param y_hat_test: predicted values
    :param freq: data frequency
    :return:

    Reference:
    https://github.com/Mcompetitions/M4-methods/blob/master/ML_benchmarks.py
    """
    y_hat_naive = []

    for i in range(freq, len(insample)):
        y_hat_naive.append(insample[(i - freq)])

    masep = jnp.mean(jnp.abs(insample[freq:] - jnp.asarray(y_hat_naive)))

    return jnp.mean(jnp.abs(y_test - y_hat_test)) / masep


def point_forecast_eval(_y, _y_star, n_train, m):

    sMAPE = smape(_y_star[n_train:], _y[n_train:])
    MASE = mase(_y[:n_train], _y[n_train:], _y_star[n_train:], m)

    return sMAPE, MASE