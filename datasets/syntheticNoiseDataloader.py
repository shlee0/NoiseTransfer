import torch
import torchvision.transforms as transforms

import numpy as np
import functools

from utils import randPos, np_crop

def get_gaussian_noisy(x, std):
	noisy = x + np.random.normal(scale=std, size=np.shape(x))
	return np.clip(noisy, 0, 1)

def get_poisson_noisy(x, chi):
	noisy = np.random.poisson(lam=x * chi, size=np.shape(x)) / chi
	return np.clip(noisy, 0, 1)

def get_poissonGaussian_noisy(x, chi, std):
	noisy = np.random.poisson(lam=x * chi, size=np.shape(x)) / chi
	noisy = noisy + np.random.normal(scale=std, size=np.shape(x))
	return np.clip(noisy, 0, 1)


class SyntheticNoiseDataloader(torch.utils.data.Dataset):
	def __init__(self, div2k_imgs, sidd_imgs, patch_size):
		sample_transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.RandomHorizontalFlip(),
			transforms.RandomChoice([
				transforms.RandomRotation((0,0)),
				transforms.RandomRotation((90,90)),
				transforms.RandomRotation((180,180)),
				transforms.RandomRotation((270,270))
			]),
			# transforms.ToTensor()
		])
		self.sample_transform = sample_transform
		
		self.div2k_imgs = div2k_imgs
		self.sidd_imgs = sidd_imgs
		self.patch_size = patch_size
		self.BLUR_KSIZE = 21


	def __getitem__(self, idx):
		# select random noise model
		rand_idx = np.random.choice(np.arange(3), p=[0.2, 0.2, 0.6])
		if rand_idx == 0:
			std = np.random.uniform(0, 70) / 255
			noise_func = functools.partial(get_gaussian_noisy, std=std)
		
		elif rand_idx == 1:
			chi = np.random.uniform(5, 100)
			noise_func = functools.partial(get_poisson_noisy, chi=chi)

		elif rand_idx == 2:
			chi = np.random.uniform(5, 100)
			std = np.random.uniform(0, 70) / 255
			noise_func = functools.partial(get_poissonGaussian_noisy, chi=chi, std=std)
		

		def _f():
			blur = False

			if np.random.rand() < 0.8: # div2k
				img_idx = np.random.randint(len(self.div2k_imgs))
				clean = self.div2k_imgs[img_idx]
			else: # sidd
				img_idx = np.random.randint(len(self.sidd_imgs))
				clean = self.sidd_imgs[img_idx]
				blur = True
					
			sample = clean
			h, w, _ = sample.shape
			top, left = randPos(h, w, self.patch_size)
			sample = np_crop(sample, top, left, self.patch_size)
			clean = self.sample_transform(sample)
			if blur:
				clean = transforms.functional.gaussian_blur(clean, self.BLUR_KSIZE)
			_clean = np.float32(clean) / 255
			noisy = noise_func(_clean).astype(np.float32)
			clean = transforms.ToTensor()(clean)
			noisy = transforms.ToTensor()(noisy)
			
			return clean, noisy

		clean, noisy = _f()
		clean2, noisy2 = _f()
		return clean, noisy, noisy2


	def __len__(self):
		return len(self.div2k_imgs)