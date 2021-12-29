class Beta(object):
	def __init__(self, a, b):
		self.a, self.b = a, b

	def pdf(self, x):
		g = self.gamma_fn(self.a + self.b) / (self.gamma_fn(self.a) + self.gamma_fn(self.b))
		return g * (x**(self.a-1)) * ((1-x)**(self.b-1))		
	
	def mean(self):
		return self.a / (self.b + self.b)

	def gamma_fn(self, n):
		c = 1
		for i in range(2, n):
			c *= i
		return c
