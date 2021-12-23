import numpy as np
import matplotlib.pyplot as plt

from distributions import Distributions as D

if __name__ == "__main__":
  data = np.random.rand(20000)

  b1 = D.binomial(5, 3, data)
  b2 = D.binomial(20, 12, data)
  b3 = D.binomial(100, 60, data)
  b4 = D.binomial(1000, 600, data)

  fig, axs = plt.subplots(4)
  axs[0].plot(data, b1, '.')
  axs[1].plot(data, b2, '.')
  axs[2].plot(data, b3, '.')
  axs[3].plot(data, b4, '.')

  plt.show()
