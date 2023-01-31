import torch
import torchvision.transforms as transforms

from .imageDataset import ImageDataset

from utils import randPos, np_crop
import numpy as np
import random


class DenoisingDataset(torch.utils.data.Dataset):
	def __init__(self, gt_paths, noisy_paths=None, patch_size=None, noise_fn=None, random_aug=True, lazy_load=False):
		self.gt_set = ImageDataset(gt_paths, lazy_load)
		self.noisy_set = None
		if noisy_paths is not None:
			self.noisy_set = ImageDataset(noisy_paths, lazy_load)

		self.patch_size = patch_size
		self.noise_fn = noise_fn
		self.random_aug = random_aug


	def __getitem__(self, idx):
		if self.noisy_set is None:
			clean = self.gt_set[idx]

			if self.patch_size is not None:
				h, w, _ = clean.shape
				top, left = randPos(h, w, self.patch_size)
				clean = np_crop(clean, top, left, self.patch_size)

			if self.random_aug:
				clean = transforms.ToPILImage()(clean)
				if random.random() < 0.5:
					clean = transforms.functional.hflip(clean)
				angle = random.choice([0,90,180,270])
				clean = transforms.functional.rotate(clean, angle)

			_clean = np.float32(clean) / 255
			noisy = self.noise_fn(_clean).astype(np.float32)

		else:
			clean = self.gt_set[idx]
			noisy = self.noisy_set[idx]

			if self.patch_size is not None:
				h, w, _ = clean.shape
				top, left = randPos(h, w, self.patch_size)
				clean = np_crop(clean, top, left, self.patch_size)
				noisy = np_crop(noisy, top, left, self.patch_size)

			if self.random_aug:
				
				clean = transforms.ToPILImage()(clean)
				noisy = transforms.ToPILImage()(noisy)
				if random.random() < 0.5:
					clean = transforms.functional.hflip(clean)
					noisy = transforms.functional.hflip(noisy)
				angle = random.choice([0,90,180,270])
				clean = transforms.functional.rotate(clean, angle)
				noisy = transforms.functional.rotate(noisy, angle)

		clean = transforms.ToTensor()(clean)
		noisy = transforms.ToTensor()(noisy)
		return clean, noisy


	def __len__(self):
		return len(self.gt_set)