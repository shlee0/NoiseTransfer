import torch
import torchvision.transforms as transforms
import random
import numpy as np

from utils import np_crop, randInt



class SIDDTrainDataset(torch.utils.data.Dataset):
	def __init__(self, gt_imgs, noisy_imgs, patch_size, iso_list, iso_imgIdx_dict):
		self.gt_imgs = gt_imgs
		self.noisy_imgs = noisy_imgs
		self.patch_size = patch_size
		self.min_cut_psize = patch_size // 4
		self.max_cut_psize = 3 * self.min_cut_psize
		self.iso_list = iso_list
		self.iso_imgIdx_dict = iso_imgIdx_dict


	def __getitem__(self, img_idx):
		def _clean_noisy_pair(idx, h_size, w_size):
			sample1 = self.gt_imgs[idx]
			sample2 = self.noisy_imgs[idx]

			h, w, _ = sample1.shape
			top = randInt(h, h_size)
			left = randInt(w, w_size)
			sample1 = np_crop(sample1, top, left, h_size, w_size)
			sample2 = np_crop(sample2, top, left, h_size, w_size)
			
			sample1 = transforms.ToPILImage()(sample1)
			sample2 = transforms.ToPILImage()(sample2)
		
			if random.random() < 0.5:
				sample1 = transforms.functional.hflip(sample1)
				sample2 = transforms.functional.hflip(sample2)

			angle = random.choice([0,90,180,270])
			sample1 = transforms.functional.rotate(sample1, angle)
			sample2 = transforms.functional.rotate(sample2, angle)

			clean = transforms.ToTensor()(sample1)
			noisy = transforms.ToTensor()(sample2)
			return clean, noisy


		def _augment(idx, alpha=None):
			clean, noisy = _clean_noisy_pair(idx, self.patch_size, self.patch_size)
			if alpha is not None: # interpolation
				noisy = alpha * noisy + (1 - alpha) * clean

			# cutmix
			if np.random.rand() < 0.5:
				cut_hsize = np.random.randint(self.min_cut_psize, self.max_cut_psize)
				cut_wsize = np.random.randint(self.min_cut_psize, self.max_cut_psize)
				clean2, noisy2 = _clean_noisy_pair(idx, cut_hsize, cut_wsize)
				if alpha is not None:
					noisy2 = alpha * noisy2 + (1 - alpha) * clean2

				top = randInt(self.patch_size, cut_hsize)
				left = randInt(self.patch_size, cut_wsize)
				clean[:, top:top+cut_hsize, left:left+cut_wsize] = clean2
				noisy[:, top:top+cut_hsize, left:left+cut_wsize] = noisy2
				

			# randomness
			noise = noisy - clean
			noise_sum = torch.sum(torch.abs(noise), dim=0)
			noise_mask = (noise_sum > 1e-3) * 1.0

			random_noise = (torch.rand(noisy.shape) - 0.5) / 255
			random_noise *= noise_mask 
			noisy += random_noise
			noisy = torch.clamp(noisy, 0.0, 1.0)
			return clean, noisy


		alpha = None
		if np.random.rand() < 0.5:
			alpha = np.random.uniform(0.8, 1)
		clean, noisy = _augment(img_idx, alpha)

		iso = self.iso_list[img_idx]
		img_idx2 = random.choice(self.iso_imgIdx_dict[iso])
		clean2, noisy2 = _augment(img_idx2, alpha)
		return clean, noisy, noisy2
		

	def __len__(self):
		return len(self.gt_imgs)