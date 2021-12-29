import numpy as np

from fstat.beta import Beta 
from fstat.bernoulli import Bernoulli 

def post(x, y, a=1, b=1):
	if 0 <= x <= 1:
		prior = Beta(a, b).pdf(x)
		like = Bernoulli(x).pmf(y)
		prob = like * prior
	else:
		prob = -np.inf
	return prob
print(post(0.8, 0.3, 2, 4))

Y = Bernoulli(np.random.rand(20)).pmf(0.7)
print(Y)
