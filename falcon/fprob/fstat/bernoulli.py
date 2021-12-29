class Bernoulli(object):
	def __init__(self, k):
		self.k = k

	def pmf(self, p):
		return (p**self.k) * (1-p)**(1-self.k)
