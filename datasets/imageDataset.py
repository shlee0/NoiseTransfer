import torch
import numpy as np
from PIL import Image
from torchvision import transforms


class ImageDataset(torch.utils.data.Dataset):
	def __init__(self, paths, lazy_load=False):
		self.lazy_load = lazy_load
		self.imgs = []
		for p in paths:
			self.imgs.append(Image.open(p))

		if not lazy_load:
			for i in range(len(self.imgs)):
				self.imgs[i] = np.array(self.imgs[i])
				

	def __getitem__(self, idx):
		if self.lazy_load:
			sample = np.array(self.imgs[idx])
		else:
			sample = self.imgs[idx]

		return sample


	def __len__(self):
		return len(self.imgs)