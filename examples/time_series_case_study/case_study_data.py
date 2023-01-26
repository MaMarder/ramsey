import haiku as hk
import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp
from ramsey.data import load_m4_time_series_data
from case_study_utils import fontsize

from jax.config import config
config.update("jax_enable_x64", True)

rng = hk.PRNGSequence(23)

(y, x), (train_idxs, test_idxs) = load_m4_time_series_data(interval='hourly')

n_series = 4
idxs_series = jnp.array([28, 32, 242, 70])

n_test = 48
n_train = 200
n = n_test + n_train

y_train = y[:, train_idxs, :]
y_train = y_train[idxs_series, -n_train:, :]
y_train_mean = y_train.mean(axis=1, keepdims=True)
y_train_std = y_train.std(axis=1, keepdims=True)

y_train = (y_train - y_train_mean) / y_train_std

y_test = y[:, test_idxs, :]
y_test = y_test[idxs_series, :n_test, :]
y_test = (y_test - y_train_mean) / y_train_std

y = jnp.concatenate([y_train, y_test], axis=1)

x = jnp.arange(n) / n_train
x = jnp.tile(x, [n_series, 1]).reshape((n_series, n, 1))
x_train, x_test = x[:, :n_train, :], x[:, n_train:, :]



# _, axes = plt.subplots(figsize=(10, 4), nrows=2, ncols=2)

# for _, (idx, ax) in enumerate(zip([0, 1, 2, 3], axes.flatten())):
#     xs = np.squeeze(x[idx, :, :])
#     ys = np.squeeze(y[idx, :, :])
#     idxs = np.argsort(xs)
#     ax.plot(xs[idxs], ys[idxs], color="black", alpha=0.5)
#     ax.scatter(xs[idxs], ys[idxs], color="red", marker="+", alpha=0.75)
#     ax.tick_params('both', labelsize=fontsize)
#     ax.set_title('Timeseries %d' % (_+1), fontsize=fontsize)
#     ax.grid()
#     ax.set_frame_on(False)
# plt.show(block=True)







