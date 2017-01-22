

class DataIterator:
  def __init__(self, generator, n_samples, dataset, chunk_size):
    self.n_samples = n_samples
    self.generator = generator
    self.dataset = dataset
    self.chunk_size = chunk_size

    if self.generator is None:
      self.n_samples = 0

  def get_n_samples(self):
    return self.n_samples

  def get_n_classes(self):
    return len(self.dataset.classes)

  def get_n_batches(self):
    self.get_n_samples()/self.chunk_size

  def get_void_labels(self):
    return [-1]

