""" Implements the memory functionalities of rextf """

import random
import numpy as np


class Memory:

  def __init__(self, size):
    self.size = size
    self.mem = np.ndarray((size, 5), dtype=object)
    self.iter = 0
    self.current_size = 0

  def remember(self, current_state, action, reward, next_state, crashed):
    self.mem[self.iter, :] = current_state, action, reward, next_state, crashed
    self.iter = (self.iter + 1) % self.size
    self.current_size = min(self.current_size + 1, self.size)

  def sample(self, n):
    n = min(self.current_size, n)
    random_idx = random.sample(list(range(self.current_size)), n)
    sample = self.mem[random_idx]
    return (np.stack(sample[:, i], axis=0) for i in range(5))
