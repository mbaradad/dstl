

class DataIterator:
  def __init__(self, generator, n_samples):
    self.n_samples = n_samples
    self.generator = generator

    if self.dataset is None:

      self.n_samples = 0

  def get_n_samples(self):
    return self.n_samples