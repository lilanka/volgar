import numpy as np
import matplotlib.pyplot as plt

from fstat.beta import Beta

_, axes = plt.subplots(2, 3, sharey=True, sharex=True)
axes = np.ravel(axes)

n_trials = [0, 1, 2, 3, 12, 180]
success = [0, 1, 1, 1, 6, 59]
data = zip(n_trials, success)

beta_params = [(0.5, 0.5), (1, 1), (10, 10)]
theta = np.linspace(0, 1, 1500)
for idx, (N, y) in enumerate(data):
	s_n = ("s" if (N > 1) else "")
	for jdx, (a_prior, b_prior) in enumerate(beta_params):
		p_theta_given_y = Beta(a_prior + y, b_prior + N -y).pdf(theta)

		axes[idx].plot(theta, p_theta_given_y, lw=4, color=viridish[jdx])
		axes[idx].set_yticks([])
		axes[idx].set_ylim(0, 12)
		axes[idx].plot(np.divide(y, N), 0, color="k", marker="o", ms=12)
		axes[idx].set_title(f"{N:4d} trials{s_n} {y:4d} success")
