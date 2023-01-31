import torch, random
import numpy as np

class InfiniteSampler(torch.utils.data.Sampler):
	def __init__(self, dataset_size, batch_size, unique):
		self.dataset_size = dataset_size
		self.batch_size = batch_size
		self.unique = unique
		if unique:
			assert (batch_size <= dataset_size)

	def __iter__(self):
		while True:
			if self.unique:
				rand_idxes = random.sample(range(self.dataset_size), self.batch_size)
			else:
				rand_idxes = np.random.randint(self.dataset_size, size=self.batch_size)
			yield rand_idxes

class InfiniteDataLoader(torch.utils.data.DataLoader):
	def __init__(self, dataset, batch_size, num_workers=2, unique=False):
		sampler = InfiniteSampler(len(dataset), batch_size, unique)
		super(InfiniteDataLoader, self).__init__(dataset, batch_sampler=sampler, num_workers=num_workers)