import matplotlib.pyplot as plt

class Utils(object):

  @classmethod
  def plot(self, x: list, y: list): 
    plt.plot(x, y, '.')
    plt.show()
